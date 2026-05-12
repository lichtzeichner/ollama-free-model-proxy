package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

type orModels struct {
	Data []struct {
		ID                   string   `json:"id"`
		ContextLength        int      `json:"context_length"`
		SupportedParameters  []string `json:"supported_parameters"`
		Architecture         struct {
			InputModalities []string `json:"input_modalities"`
		} `json:"architecture"`
		TopProvider          struct {
			ContextLength int `json:"context_length"`
		} `json:"top_provider"`
		Pricing struct {
			Prompt     string `json:"prompt"`
			Completion string `json:"completion"`
			Image      string `json:"image"`
		} `json:"pricing"`
	} `json:"data"`
}

// supportsToolUse checks if a model supports tool use by looking for "tools" in supported_parameters
func supportsToolUse(supportedParams []string) bool {
	for _, param := range supportedParams {
		if param == "tools" || param == "tool_choice" {
			return true
		}
	}
	return false
}

func supportsImageInput(inputModalities []string) bool {
	for _, modality := range inputModalities {
		if modality == "image" {
			return true
		}
	}
	return false
}

func isZeroPrice(price string) bool {
	return price == "" || price == "0"
}

func freeModelCachePath(basePath string) string {
	suffixes := []string{}
	if strings.ToLower(os.Getenv("TOOL_USE_ONLY")) == "true" {
		suffixes = append(suffixes, "tools")
	}
	if strings.ToLower(os.Getenv("VISION_ONLY")) == "true" {
		suffixes = append(suffixes, "vision")
	}
	if len(suffixes) == 0 {
		return basePath
	}
	return basePath + "." + strings.Join(suffixes, ".")
}

func fetchFreeModels(apiKey string) ([]string, error) {
	req, err := http.NewRequest("GET", "https://openrouter.ai/api/v1/models", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %s", resp.Status)
	}
	var result orModels
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	// Check if tool use filtering is enabled
	toolUseOnly := strings.ToLower(os.Getenv("TOOL_USE_ONLY")) == "true"
	visionOnly := strings.ToLower(os.Getenv("VISION_ONLY")) == "true"
	
	type item struct {
		id  string
		ctx int
	}
	var items []item
	for _, m := range result.Data {
		if isZeroPrice(m.Pricing.Prompt) && isZeroPrice(m.Pricing.Completion) {
			// If tool use filtering is enabled, skip models that don't support tools
			if toolUseOnly && !supportsToolUse(m.SupportedParameters) {
				continue
			}

			if visionOnly {
				if !supportsImageInput(m.Architecture.InputModalities) {
					continue
				}
				if !isZeroPrice(m.Pricing.Image) {
					continue
				}
			}
			
			ctx := m.TopProvider.ContextLength
			if ctx == 0 {
				ctx = m.ContextLength
			}
			items = append(items, item{id: m.ID, ctx: ctx})
		}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].ctx > items[j].ctx })
	models := make([]string, len(items))
	for i, it := range items {
		models[i] = it.id
	}
	return models, nil
}

func ensureFreeModelFile(apiKey, path string) ([]string, error) {
	const cacheMaxAge = 24 * time.Hour // Refresh cache daily

	if stat, err := os.Stat(path); err == nil {
		// Check if cache is still fresh
		if time.Since(stat.ModTime()) < cacheMaxAge {
			data, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			var models []string
			for _, line := range strings.Split(string(data), "\n") {
				line = strings.TrimSpace(line)
				if line != "" {
					models = append(models, line)
				}
			}
			return models, nil
		}
		// Cache is stale, will fetch fresh models below
	}

	// Fetch fresh models from API
	models, err := fetchFreeModels(apiKey)
	if err != nil {
		// If fetch fails but we have a cached file (even if stale), use it
		if _, statErr := os.Stat(path); statErr == nil {
			data, readErr := os.ReadFile(path)
			if readErr == nil {
				var cachedModels []string
				for _, line := range strings.Split(string(data), "\n") {
					line = strings.TrimSpace(line)
					if line != "" {
						cachedModels = append(cachedModels, line)
					}
				}
				return cachedModels, nil
			}
		}
		return nil, err
	}

	// Save fresh models to cache
	_ = os.WriteFile(path, []byte(strings.Join(models, "\n")), 0644)
	return models, nil
}
