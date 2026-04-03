# Contributing to RoboBridge

RoboBridge에 기여해 주셔서 감사합니다! 이 문서는 프로젝트에 기여하기 위한 가이드라인을 설명합니다.

Thank you for your interest in contributing to RoboBridge! This document describes guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Commit Message Convention](#commit-message-convention)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

- 모든 기여자를 존중하고 건설적인 피드백을 제공합니다.
- 차별적이거나 공격적인 언어를 사용하지 않습니다.
- 기술적 논의에 집중하고 개인 공격을 삼가합니다.

---

## Getting Started

### Ways to Contribute

1. **Bug Reports** - 버그를 발견하면 Issue를 생성해 주세요
2. **Feature Requests** - 새로운 기능 제안은 Issue로 논의해 주세요
3. **Code Contributions** - 버그 수정, 기능 추가, 성능 개선
4. **Documentation** - 문서 개선, 오타 수정, 예제 추가
5. **Testing** - 테스트 커버리지 향상

### Before You Start

- 기존 Issue를 확인하여 중복 작업을 피해주세요
- 큰 변경사항은 먼저 Issue에서 논의해 주세요
- 작은 수정(오타, 포맷팅)은 바로 PR을 보내도 됩니다

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/robobridge.git
cd robobridge

# Add upstream remote
git remote add upstream https://github.com/PRISM-SKKU/robobridge.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run specific test file
PYTHONPATH=src pytest tests/unit/test_types.py -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=src/robobridge --cov-report=html
```

---

## Commit Message Convention

[Conventional Commits](https://www.conventionalcommits.org/) 규격을 따릅니다.

### Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description | SemVer |
|------|-------------|--------|
| `feat` | 새로운 기능 추가 | MINOR |
| `fix` | 버그 수정 | PATCH |
| `docs` | 문서 변경 | - |
| `style` | 코드 포맷팅 (기능 변경 없음) | - |
| `refactor` | 리팩토링 (기능 변경 없음) | - |
| `perf` | 성능 개선 | PATCH |
| `test` | 테스트 추가/수정 | - |
| `chore` | 빌드, 설정 파일 변경 | - |
| `ci` | CI/CD 설정 변경 | - |

### Scope (Optional)

변경이 영향을 미치는 모듈을 명시합니다:

- `perception`, `planner`, `controller`, `robot`, `monitor`
- `core`, `client`, `config`, `cli`
- `docs`, `tests`

### Examples

```bash
# 새로운 기능
feat(planner): add support for Anthropic Claude models

# 버그 수정
fix(perception): handle empty depth image gracefully

# 문서 수정
docs: update installation guide for Windows

# 리팩토링
refactor(core): simplify adapter connection logic

# Breaking change (! 사용)
feat(config)!: change default topic namespace from /craf to /robobridge

BREAKING CHANGE: All topic names have been updated. 
Update your config files accordingly.

# 여러 줄 본문
fix(robot): prevent command timeout on slow networks

Increased default timeout from 5s to 15s.
Added exponential backoff for retry logic.

Closes #42
```

### Rules

1. **제목은 50자 이내**로 작성
2. **제목은 명령형**으로 작성 (Add, Fix, Update, Remove)
3. **제목 끝에 마침표 없음**
4. 본문은 **72자**에서 줄바꿈
5. 본문에는 **"왜"**를 설명 (무엇을 했는지는 코드가 말해줌)
6. Issue 참조: `Closes #123`, `Fixes #456`, `Refs #789`

---

## Pull Request Process

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feat/your-feature-name
```

### Branch Naming

```
<type>/<short-description>

Examples:
- feat/add-ollama-provider
- fix/perception-memory-leak
- docs/update-quickstart
- refactor/simplify-config-loader
```

### 2. Make Changes

- 하나의 PR은 **하나의 목적**만 가지도록
- 관련 없는 변경사항은 별도 PR로 분리
- 테스트 추가/수정 포함

### 3. Submit PR

```bash
git push -u origin feat/your-feature-name
```

GitHub에서 PR을 생성하고 아래 템플릿을 따라주세요:

```markdown
## Summary
<!-- 변경사항을 간단히 설명 -->

## Changes
- 
- 

## Related Issues
<!-- Closes #123 -->

## Checklist
- [ ] Tests pass locally (`pytest tests/`)
- [ ] Code follows project style
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow convention
```

### 4. Review Process

1. CI 테스트 통과 확인
2. 리뷰어 피드백 반영
3. Approve 후 Squash and Merge

### PR Title Convention

PR 제목도 Conventional Commits 형식을 따릅니다:

```
feat(planner): add support for Google Gemini
fix(monitor): resolve race condition in feedback loop
docs: add Korean translation for quickstart
```

---

## Code Style

### Python

- **Formatter**: `black` (line-length: 100)
- **Linter**: `ruff`
- **Type hints**: 모든 public API에 필수

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check (optional)
mypy src/robobridge
```

### Style Guidelines

```python
# Good: Type hints, docstring for public API
def detect(
    self,
    rgb: np.ndarray,
    depth: Optional[np.ndarray] = None,
    object_list: Optional[List[str]] = None,
) -> List[Detection]:
    """Detect objects in the image."""
    ...

# Bad: No type hints, unclear naming
def detect(self, img, d=None, objs=None):
    ...
```

### Import Order

```python
# 1. Standard library
import json
import logging
from typing import List, Optional

# 2. Third-party
import numpy as np
from langchain_core.messages import HumanMessage

# 3. Local
from robobridge.modules.base import BaseModule
from .types import Detection
```

---

## Testing

### Test Structure

```
tests/
├── unit/           # 단위 테스트 (모킹 사용)
│   ├── test_types.py
│   ├── test_config.py
│   └── ...
└── integration/    # 통합 테스트 (실제 모듈 연동)
    ├── test_pipeline.py
    └── test_client.py
```

### Writing Tests

```python
import pytest
from robobridge.modules.planner.types import Plan, PlanStep

class TestPlan:
    def test_plan_creation(self):
        """Plan should be created with valid steps."""
        step = PlanStep(step_id=0, skill="pick", target_object="cup")
        plan = Plan(plan_id="plan_001", instruction="Pick the cup", steps=[step])
        
        assert plan.plan_id == "plan_001"
        assert len(plan.steps) == 1

    def test_plan_empty_steps_raises(self):
        """Plan with no steps should raise ValueError."""
        with pytest.raises(ValueError):
            Plan(plan_id="plan_001", instruction="Do something", steps=[])
```

### Test Naming

```python
def test_<what>_<condition>_<expected>():
    """<What> should <expected> when <condition>."""
    
# Examples:
def test_perception_empty_image_returns_empty_list():
def test_planner_invalid_instruction_raises_error():
def test_controller_generates_trajectory_for_pick_skill():
```

---

## Documentation

### Docstrings

Google style docstrings를 사용합니다:

```python
def process(
    self,
    instruction: str,
    object_poses: Optional[List[Dict]] = None,
) -> Plan:
    """Generate a task plan from natural language instruction.

    Args:
        instruction: Natural language command (e.g., "Pick up the red cup")
        object_poses: List of detected objects with their poses

    Returns:
        Plan object containing executable steps

    Raises:
        PlannerError: If plan generation fails

    Example:
        >>> planner = Planner(provider="openai", model="gpt-4o")
        >>> plan = planner.process("Pick up the cup")
        >>> print(plan.steps[0].skill)
        'pick'
    """
```

### Documentation Files

- `docs/` 폴더의 마크다운 파일들
- 영어와 한국어 버전 모두 유지 (`*.md`, `*.ko.md`)
- MkDocs로 빌드: `mkdocs serve`

---

## Questions?

- Issue를 생성하거나
- Discussion에서 질문해 주세요

감사합니다! 🚀
