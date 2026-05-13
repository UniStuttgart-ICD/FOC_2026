# Robots and Embodied Agents in Collaborative Construction

## Session Log

Date: 2026-05-11

## Starting Brief

This project is for a workshop where users create their own collaborative agent for human-machine collaboration.

The core activity is collaborative construction: a human, a robot, and an agent work together to assemble a simple timber structure from rectangular timber beams around 30-50 cm long.

The conceptual question is whether the agent is embodied in the robot or acts as a separate participant. Initial direction: start with the agent as a separate persona, not embodied in the robot.

Participants will design both:

- the agent personality and behavior;
- the physical construction strategy for the timber assembly.

Each group invents its own structure. There is no fixed target object for everyone to reproduce. The current goal is to create one reference scenario and use it to design and think through uncertainty in the collaborative process.

## Scenario Drafts

Three complete scenario drafts expand the initial approaches:

- [Scenario 1: Uncertainty Ladder](scenario-1-uncertainty-ladder.md)
- [Scenario 2: Open Breakdown](scenario-2-open-breakdown.md)
- [Scenario 3: Designed Friction Cards](scenario-3-designed-friction-cards.md)
- [Scenario 3 Expanded: Act, Agent Attributes, and Construction Problem Cards](scenario-3-act-workflow-expanded.md)

## Conceptual Framework From Image

The image frames three actors:

- Human: perceives, decides, handles materials, and responds to unexpected site conditions.
- Robot: provides physical manipulation, reach, repeatability, and constrained motion.
- Agent: mediates perception, planning, dialogue, uncertainty handling, and coordination.

The shared object is a timber assembly on a work surface. The interesting design space is not only task execution, but how the three actors negotiate, recover, and adapt when the build does not go as planned.

## Project Context

The existing `pipecat-agent` project is a voice robot agent. Its architecture already separates:

- Voice Runtime: wake word, STT, TTS, audio pipeline, and turn timing.
- Agent Control: LangChain/LangGraph orchestration, prompt, dialogue, and tool choice.
- Robot Control: task policy, validation, robot context, and MoveIt execution.

Current robot persona is `Mave`, an embodied UR10 arm. For the workshop scenario, a key conceptual change is to let the agent be its own actor, separate from the robot body.

Relevant existing affordances:

- Voice command workflow with wake word.
- MoveIt-backed robot action planning and execution.
- Robot context and fresh observation requirements.
- User sensing hooks such as gaze, user position, and manual targets.
- Prompt parts for persona, embodiment, robot contract, and uncertainty-aware speech.

## Working Hypothesis

The workshop can frame uncertainty as the main site of collaboration. The build should be simple enough to finish physically, but rich enough that small disruptions require interpretation, dialogue, and role negotiation.

The agent should not simply command the robot. It should act as a collaborative coordinator that keeps track of the plan, names the observable conflict when uncertainty appears, asks for the missing judgement or measurement, and proposes next actions with clear actor assignments.

## Uncertainty Writing Rule

Uncertainty must be described as a concrete event, not a general condition.

Every uncertainty scenario should answer:

- What was the planned next action?
- What exactly happened in the physical setup or dialogue?
- What observable evidence shows that the plan no longer works?
- What must the robot stop doing or change doing?
- What must the human inspect, decide, hold, move, or confirm?
- What does the agent say at the moment of uncertainty?
- How does the final structure change because of the event?

Avoid vague phrases such as "choose between alternatives" unless the alternatives are named with physical consequences.

## Initial Scenario Seed

Three actors assemble a small timber structure from short rectangular beams:

- A participant-invented assembly, with one reference scenario developed first.
- Possible reference object: a base frame or small bridge-like structure.
- Current reference deck constraint: exactly six timber elements total.
- Six-element kit: two triangular base rails, two columns, one crown beam, and one movable strut.
- Uncertainty should force reinterpretation of those six pieces, not the addition of spare timber.
- Beams are placed on a table within robot reach.
- The human can hold, align, hand over, inspect, and decide.
- The robot can pick, place, press, or indicate positions.
- The agent coordinates the sequence and handles uncertainty through dialogue.

## Initial Uncertainty Seeds

- A beam is missing, swapped, warped, too short, or not where expected.
- The robot cannot reach the intended placement.
- The human places a part differently than planned.
- The structure becomes unstable during assembly.
- The agent has stale or ambiguous sensing, such as unclear gaze or "put it there".
- The robot plan fails or the gripper cannot securely handle the beam.
- Human and robot disagree implicitly through action: the human holds one part while the robot prepares for another.

Uncertainty should include all three families:

- Material uncertainty: wrong, missing, warped, unstable, or hard-to-grip timber.
- Spatial uncertainty: ambiguous placement, alignment, reachability, handover, or support.
- Role uncertainty: who decides, who acts, who verifies, and who adapts when the plan changes.

## Emerging Design Direction

Start with the agent as a third actor:

- Name and personality should make it feel present but not pretend to be the robot.
- Relationship to robot: collaborator, interpreter, and safety-aware coordinator.
- Relationship to human: workshop partner, not supervisor.
- Relationship to uncertainty: calm, explicit, and generative. It should turn breakdowns into collaborative choices.

Workshop development now includes a single-file slide deck with a rotatable Three.js model of the six-element reference act.

## Open Questions

- Should uncertainty be staged by facilitators, emerge naturally from sensing/manipulation limits, or both?
- How visible should the agent be: voice only, screen/avatar, AR overlay, or text trace?
- Should the three actors share one plan, or should each maintain a different partial understanding?
- Should the robot be allowed to propose actions, or only execute after human-agent agreement?
- What should count as success: completed structure, graceful recovery, rich reflection, or all three?
