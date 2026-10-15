# This file contains convenience targets for the project.
# It is not intended to be used as a build system.
# See the README for more information.

.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests with basic environment setup"
	@echo "  test-race      - Run tests with race detection"
	@echo "  test-cover     - Run tests with coverage reporting"
	@echo "  test-record    - Run tests with re-recording of httprr files"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Check line lengths and run golangci-lint"
	@echo "  fix            - Auto-fix lint issues (alias: lint-fix); 'make fix lint' to fix then re-check"
	@echo "  lint-testing   - Check test patterns and practices (httprr, etc.)"
	@echo "  lint-testing-fix - Check and attempt to fix test patterns"
	@echo "  lint-architecture - Check architectural rules and patterns"
	@echo ""
	@echo "Other:"
	@echo "  build-examples - Build all example projects to verify they compile"
	@echo "  update-examples - Update langchaingo version in all examples"
	@echo "  docs           - Generate documentation"
	@echo "  clean          - Clean lint cache"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Git Hooks:"
	@echo "  pre-push       - Run lint and fast tests (suitable for git pre-push hook)"
	@echo "  install-git-hooks  - Install git hooks (sets up pre-push hook)"

.PHONY: test
test:
	DOCKER_HOST=$$(docker context inspect -f='{{.Endpoints.docker.Host}}' 2>/dev/null || echo "unix:///var/run/docker.sock") \
	TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="/var/run/docker.sock" \
	go test ./...

# Lint tooling is pinned in tools.mod (same setup and versions as the runreveal repo).
GO_TOOL := go tool -modfile=tools.mod
GOLANG_CI := $(GO_TOOL) golangci-lint
GOLINES := $(GO_TOOL) golines
GO_FILES := find . -name "*.go" -not -path "./examples/*" -not -path "./docs/*" -print0

.PHONY: lint
lint: fmtcheck
	$(GOLANG_CI) run --timeout 5m ./...

.PHONY: fmtcheck
# Any files with really long lines?
fmtcheck:
	! $(GO_FILES) | xargs -0 -P 8 -I {} $(GOLINES) --base-formatter="gofmt" --max-len 128 -l {} | grep .

# Auto-fix what can be fixed: wrap long lines, then apply golangci-lint fixes.
# Usage: make fix (or make fix lint to fix and then re-check)
.PHONY: fix lint-fix
fix lint-fix:
	$(GO_FILES) | xargs -0 -P 8 -I {} $(GOLINES) --base-formatter="gofmt" --max-len 128 -w {}
	$(GOLANG_CI) run --fix --timeout 5m ./...

.PHONY: lint-exp
lint-exp:
	$(GOLANG_CI) run --fix --config .golangci-exp.yaml ./...

.PHONY: lint-all
lint-all: lint

.PHONY: docs
docs:
	@echo "Generating documentation..."
	$(MAKE) -C docs build

.PHONY: test-race
test-race:
	DOCKER_HOST=$$(docker context inspect -f='{{.Endpoints.docker.Host}}' 2>/dev/null || echo "unix:///var/run/docker.sock") \
	TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="/var/run/docker.sock" \
	go test -race ./...

.PHONY: test-cover
test-cover:
	DOCKER_HOST=$$(docker context inspect -f='{{.Endpoints.docker.Host}}' 2>/dev/null || echo "unix:///var/run/docker.sock") \
	TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="/var/run/docker.sock" \
	go test -cover ./...

.PHONY: test-record
test-record:
	@echo "Re-recording HTTP interactions for all packages using httprr..."
	@echo "Note: Running with limited parallelism to avoid API rate limits"
	PACKAGES=$$(go run ./internal/devtools/rrtool list-packages -format=paths) && \
	echo "Recording HTTP interactions for packages:" && \
	echo "$$PACKAGES" | tr ' ' '\n' | sed 's/^/  /' && \
	echo "" && \
	env DOCKER_HOST=$$(docker context inspect -f='{{.Endpoints.docker.Host}}' 2>/dev/null || echo "unix:///var/run/docker.sock") \
	TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="/var/run/docker.sock" \
	go test $$PACKAGES -httprecord=. -httprecord-delay=1s -p 2 -parallel=2 -timeout=300s


.PHONY: run-pkgsite
run-pkgsite:
	go run golang.org/x/pkgsite/cmd/pkgsite@latest

.PHONY: clean
clean: clean-lint-cache

.PHONY: clean-lint-cache
clean-lint-cache:
	$(GOLANG_CI) cache clean

.PHONY: build-examples
build-examples:
	for example in $(shell find ./examples -mindepth 1 -maxdepth 1 -type d); do \
		(cd $$example; echo Build $$example; go mod tidy; go build -o /dev/null) || exit 1; done

.PHONY: update-examples
update-examples:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Usage: make update-examples VERSION=v0.1.14-pre.1"; \
		exit 1; \
	fi
	@echo "Updating examples to $(VERSION)..."
	@go run ./internal/devtools/examples-updater -version $(VERSION)

.PHONY: add-go-work
add-go-work:
	go work init .
	go work use -r .

.PHONY: lint-devtools
lint-devtools:
	go run ./internal/devtools/lint -v

.PHONY: lint-devtools-fix
lint-devtools-fix:
	go run ./internal/devtools/lint -fix -v

.PHONY: lint-architecture
lint-architecture:
	go run ./internal/devtools/lint -architecture -v

.PHONY: lint-prepush
lint-prepush:
	go run ./internal/devtools/lint -prepush -v

.PHONY: lint-prepush-fix
lint-prepush-fix:
	go run ./internal/devtools/lint -prepush -fix -v

.PHONY: lint-testing
lint-testing:
	go run ./internal/devtools/lint -testing -v

.PHONY: lint-testing-fix
lint-testing-fix:
	go run ./internal/devtools/lint -testing -fix -v

.PHONY: pre-push
pre-push:
	@echo "Running pre-push checks..."
	@$(MAKE) lint
	@go test -short ./...
	@echo "✅ Pre-push checks passed!"

.PHONY: install-git-hooks
install-git-hooks:
	@./internal/devtools/git-hooks/install-git-hooks.sh
