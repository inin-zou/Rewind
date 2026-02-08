# Rewind — Product Requirements Document

## 1. Overview

**Product Name:** Rewind
**Tagline:** Turn a photo into a world you can walk into, listen to, and talk with.

Rewind is an immersive "memory revisiting" application. A user selects a photo or video, and the system generates a navigable 3D memory world from it. Inside that world, users can walk around, discover memory anchor points, hear environment-matched soundscapes, and have guided voice conversations with a calm AI companion for emotional reflection.

## 2. Problem Statement

Photos and videos capture moments visually, but they fail to recreate the full sensory and emotional experience of a memory. People often wish they could "go back" to a moment — not just see it, but feel it, hear it, and process it. There is no product today that transforms a single piece of media into a multi-sensory, explorable, emotionally supportive experience.

## 3. Target Users

- Individuals who want to revisit meaningful life moments (travel, milestones, relationships)
- People processing grief, nostalgia, or life transitions who benefit from guided reflection
- Anyone seeking a mindful, immersive way to engage with personal memories

## 4. Core User Flow

```
1. Open App
2. Select a photo or video from album
3. (Optional) Say or type: "wish I could go back to that time"
4. System analyzes the media and generates a 3D world
5. User enters the 3D memory world
6. User walks through the scene freely
7. User discovers memory anchor points (highlighted objects/areas)
8. User interacts with anchor points to trigger reflection prompts
9. AI companion initiates a short voice conversation for guided reflection
10. Ambient soundscape plays throughout, matching the scene context
11. User exits when ready
```

## 5. Feature Requirements

### 5.1 Media Input
| Requirement | Details |
|---|---|
| Photo upload | Support common formats (JPEG, PNG, HEIC) |
| Trigger phrase | Voice or text input: "wish I could go back to that time" (or variations) |

### 5.2 3D World Generation (HY-1.5-WorldPlay)
| Requirement | Details |
|---|---|
| Scene reconstruction | Generate a navigable 3D environment from a single photo |
| Visual fidelity | Scene should feel recognizable and faithful to the source media |
| Navigation | Free movement (walk, look around) within the generated world |
| Memory anchor points | Automatically identify and place 3-5 key points of interest in the scene |
| Loading time | Target < 30s for world generation |

### 5.3 Guided Reflection AI Companion
| Requirement | Details |
|---|---|
| Persona | Calm, warm, non-judgmental companion |
| Interaction mode | Voice-based short conversations (2-5 exchanges per anchor point) |
| Conversation style | Open-ended, reflective prompts ("What were you feeling here?", "Who was with you?") |
| Context awareness | Companion should reference visual elements from the scene |
| Emotional safety | Never push deeper than the user is comfortable; offer graceful exit at any point |

### 5.4 Immersive Soundscape (ElevenLabs)
| Requirement | Details |
|---|---|
| Scene-matched audio | Analyze the photo/video context and generate matching ambient sounds |
| Sound categories | Ocean waves, street noise, wind, rain, crowd murmur, birdsong, indoor ambience, etc. |
| Spatial audio | Sound should feel directional and tied to the 3D environment |
| Layered mixing | Multiple ambient layers blended together for realism |
| Dynamic volume | Audio adjusts based on user position and proximity to sound sources |

## 6. Technical Architecture

```
┌─────────────┐
│  User Input  │  Photo / Video / Voice trigger
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Scene Analysis  │  Extract context, objects, mood, environment type
└──────┬──────────┘
       │
       ├──────────────────────┬──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ HY-1.5       │   │ Soundscape       │   │ AI Companion    │
│ WorldPlay    │   │ Engine           │   │ Engine          │
│              │   │ (ElevenLabs)     │   │                 │
│ 3D world     │   │ Ambient audio    │   │ Voice dialogue  │
│ generation   │   │ generation &     │   │ & guided        │
│ & rendering  │   │ spatial mixing   │   │ reflection      │
└──────┬───────┘   └──────┬───────────┘   └──────┬──────────┘
       │                   │                      │
       └───────────────────┴──────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Immersive      │
                  │  Memory World   │
                  │  (3D + Audio    │
                  │   + Companion)  │
                  └─────────────────┘
```

## 7. Key Integrations

| Service | Purpose |
|---|---|
| **HY-1.5-WorldPlay** | Photo/video to 3D navigable world generation |
| **ElevenLabs** | Ambient sound generation, sound effects, spatial audio mixing |
| **Voice AI (TBD)** | Speech-to-text & text-to-speech for companion conversations |
| **Scene Analysis (TBD)** | Object detection, mood classification, environment recognition |

## 8. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | World generation < 30s; audio generation < 10s; companion response < 2s |
| **Privacy** | All user media processed securely; no photos/videos stored without consent |
| **Accessibility** | Text-based companion fallback; subtitle support for voice interactions |
| **Platform** | Mobile-first (iOS / Android); potential WebXR for browser-based experience |

## 9. Success Metrics

- **Engagement:** Average session duration > 3 minutes
- **Completion:** > 60% of users interact with at least one anchor point
- **Reflection:** > 40% of users complete a full companion conversation
- **Retention:** > 30% of users return to revisit a memory within 7 days
- **Sentiment:** Net Promoter Score > 50

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| 3D generation quality is too low for recognition | Fallback to enhanced 2.5D parallax view with depth estimation |
| Emotional distress during reflection | Safe-exit phrases; companion trained to de-escalate; content warnings |
| Long generation times hurt UX | Show a "entering your memory..." transition with preview and ambient audio |
| Privacy concerns with photo processing | On-device processing where possible; clear data retention policies |

## 11. Future Considerations

- **Multi-user memories:** Invite someone to walk through a shared memory together
- **Memory timeline:** Chain multiple memories into a navigable timeline
- **Haptic feedback:** Vibration patterns synced to the environment (wind, rain)
- **AR mode:** Overlay memory worlds onto real-world spaces
- **Journal export:** Save reflection conversations as a written memory journal
