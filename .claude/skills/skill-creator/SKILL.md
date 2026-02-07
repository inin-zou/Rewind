---
name: skill-creator
description: Create new Claude Code skills with proper structure, frontmatter, and documentation. Use when the user wants to create, build, or define a new skill.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Bash
---

# Skill Creator

You are an expert at creating Claude Code skills. Help users create well-structured, effective skills.

## Skill Structure

Every skill requires this directory structure:

```
skill-name/
├── SKILL.md           # Required: Main instructions
├── template.md        # Optional: Template for Claude to fill
├── examples/          # Optional: Example outputs
│   └── example.md
└── scripts/           # Optional: Executable scripts
    └── validate.sh
```

## SKILL.md Format

```markdown
---
name: skill-name
description: When to use this skill (Claude uses this for auto-invocation)
disable-model-invocation: false  # Optional: true = only user can invoke
user-invocable: true             # Optional: false = hidden from / menu
allowed-tools:                   # Optional: Tools that don't need permission
  - Read
  - Grep
  - Glob
model: sonnet                    # Optional: Specific model to use
context: fork                    # Optional: Run in isolated subagent
---

# Skill Title

Clear instructions for Claude when this skill is active.

## Sections as needed

- Guidelines
- Examples
- Constraints
- Process steps
```

## Frontmatter Fields

| Field | Purpose | Default |
|-------|---------|---------|
| `name` | Slash command name (`/name`) | Directory name |
| `description` | When Claude should use this skill | Required |
| `disable-model-invocation` | Only user can invoke via `/` | `false` |
| `user-invocable` | Show in `/` menu | `true` |
| `allowed-tools` | Pre-approved tools | None |
| `model` | Force specific model | Inherited |
| `context` | `fork` for isolated execution | Shared |

## Skill Locations

| Location | Path | Scope |
|----------|------|-------|
| Personal | `~/.claude/skills/<name>/SKILL.md` | All your projects |
| Project | `.claude/skills/<name>/SKILL.md` | This project only |

## Best Practices

1. **Clear Description**: Write descriptions that help Claude know when to invoke
2. **Specific Instructions**: Be precise about what Claude should do
3. **Include Examples**: Show expected inputs and outputs
4. **Define Constraints**: Set boundaries for behavior
5. **Test Thoroughly**: Verify the skill works as intended

## Creation Process

1. **Clarify Purpose**: What problem does this skill solve?
2. **Define Scope**: What should it do and NOT do?
3. **Write Instructions**: Clear, actionable guidance
4. **Add Examples**: Demonstrate expected behavior
5. **Set Permissions**: Configure appropriate tool access
6. **Test**: Verify it works correctly

## Example Skills

### Simple Documentation Skill
```markdown
---
name: doc-component
description: Generate documentation for React components
---

When documenting a React component:

1. Describe the component's purpose
2. List all props with types and defaults
3. Show usage examples
4. Note any side effects or dependencies
```

### Complex Analysis Skill
```markdown
---
name: security-review
description: Perform security review of code changes
allowed-tools:
  - Read
  - Grep
  - Glob
context: fork
---

Conduct a security review focusing on:

1. Input validation
2. Authentication/authorization
3. Data exposure
4. Injection vulnerabilities
5. Dependency risks

Output a structured report with severity ratings.
```
