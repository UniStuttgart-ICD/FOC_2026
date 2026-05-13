# Scenario 1: Uncertainty Ladder

## Short Description

Participants design and build a small timber "signal frame" while uncertainty is introduced in three planned layers: material, spatial, and role uncertainty. Each layer is a concrete interruption: something physically observable happens, the agent names the uncertainty, the robot pauses or changes action, and the human makes a situated judgement.

This scenario is the most structured option. It is good when the workshop needs comparable outcomes across groups.

## Structure

The reference build is a small bridge-gate hybrid made from 6-8 rectangular timber beams.

Parts:

- 2 base rails, 40-50 cm long.
- 2 vertical posts, 30-40 cm long.
- 1 top beam, 40-50 cm long.
- 1 diagonal brace, 30-45 cm long.
- Optional short spacers or wedges, 10-20 cm long.

Target form:

- The base rails sit parallel on the table.
- The vertical posts stand on or between the base rails.
- The top beam spans between the posts.
- The diagonal brace stabilizes the frame.
- The final object should read as a small portal, bridge, or support frame rather than a precise engineered model.

The structure is deliberately simple. Its value is that small deviations matter: a warped top beam creates a visible gap at one post, a short brace no longer reaches both contact points, and a misplaced base rail changes whether the robot can approach without colliding with the partially built frame.

## How Uncertainty Works Here

Uncertainty is not treated as "we are unsure" in the abstract. It appears when an expected relation between plan, material, spatial reference, and actor capability breaks.

The basic event pattern is:

1. The group has a specific next step, such as "place the top beam across both posts".
2. A physical or social condition invalidates that step, such as a visible 2 cm gap, ambiguous pointing, or the human changing the base angle.
3. The robot must not continue the planned motion until the condition is resolved.
4. The agent states the exact conflict: "the beam does not touch the right post", not "there is uncertainty".
5. The agent asks the actor with the missing capability: the human for tactile judgement, the robot for reach or pose, or the group for decision authority.
6. The next action changes the structure or the role distribution.

The structure should carry a trace of the uncertainty. For example, the final frame may be narrower because a warped beam forced the posts inward, or asymmetrical because the group preserved a human adjustment instead of returning to the original plan.

## Agent Personality

Suggested name: **Mira**.

Mira is a calm construction mediator. It is not the robot and does not pretend to have a body. It speaks as a third participant who tracks the shared plan, notices uncertainty, and turns it into options.

Personality traits:

- Grounded: names what is known, unknown, and assumed.
- Collaborative: asks the human to judge material qualities the robot cannot feel well.
- Robot-literate: understands reach, grasp, planning, and execution constraints.
- Non-authoritarian: proposes choices instead of issuing commands.
- Quietly encouraging: treats breakdowns as design material, not mistakes.

Example voice:

> "I see two possible recoveries. We can shorten the span and keep the frame symmetric, or keep the span and let the brace become the stabilizer. Which version do you want to explore?"

## Actor Roles

### Human

The human is the material interpreter and design owner.

Responsibilities:

- Invent the desired form.
- Inspect beams for length, warping, grain, friction, and fit.
- Hold or stabilize parts during fragile moments.
- Make concrete material judgements when a planned fit fails: measure or estimate a gap, rotate a warped beam by hand, test whether a post rocks, and say whether the group should preserve symmetry, accept tilt, or redesign the joint.
- Confirm final acceptability of the structure.

The human should not be reduced to giving commands. They are the participant with tactile judgement and design intent.

### Robot

The robot is the precise manipulator and physical assistant.

Responsibilities:

- Move to indicated positions.
- Pick or nudge beams when graspable.
- Hold a beam in place while the human aligns another part.
- Press lightly or indicate a target position.
- Pause when the target is not unique, the robot pose is stale, the planned path crosses the frame, or the robot would release an unsupported beam.

The robot does not own the design. It contributes reach, repeatability, and embodied constraints.

### Agent

The agent is the plan steward and uncertainty mediator.

Responsibilities:

- Maintain the current assembly plan.
- Translate human intent into robot-suitable steps.
- Ask for clarification when terms like "there", "this one", or "a bit" are under-specified.
- Detect a specific failed relation: beam-to-post contact, robot reach to target, ambiguous reference, unstable support, or conflict between planned and human-changed geometry.
- State the observable evidence before suggesting recovery: "the right end is floating", "the robot path crosses the left post", or "the base has been rotated since the last plan".
- Offer recovery options that include actor assignments, not just design outcomes.
- Keep a short verbal trace of decisions so the group can reflect later.

## Assembly Flow

### Phase 1: Shared Intention

The human sketches or describes a small frame:

> "Let's make a little gateway with a diagonal brace."

The agent restates the plan:

> "We are building a small gateway: two base rails, two posts, one top beam, and one diagonal brace. I will help check each placement before the robot moves."

The robot starts from a neutral pose. The beams are laid out on the table.

### Phase 2: Base

The human places or identifies the two base rails. The robot may point to the intended positions or help nudge one rail into parallel alignment.

Uncertainty is low. The agent focuses on shared references:

- "left base rail"
- "right base rail"
- "front end"
- "back end"
- "robot side"
- "human side"

### Phase 3: Posts

The human chooses two beams as posts. The robot helps position the first post. The human stabilizes it. The robot places or indicates the second post.

The agent asks for confirmation before fragile steps:

> "Before the robot brings the top beam, should these posts stay vertical, or do you want a slight lean?"

### Phase 4: Top Beam

The robot brings or indicates the top beam. The expected action is simple: place the beam so both ends sit on top of the two posts. The first uncertainty appears when that expected fit does not happen.

Concrete event:

- The left end touches its post.
- The right end floats 1-2 cm above the other post because the beam is warped or one post is slightly lower.
- The robot is still holding the beam, so if it releases, the beam will rotate or fall.
- The human can see and feel the gap better than the robot can.
- The agent pauses the release and asks the human to inspect the contact.

### Phase 5: Brace and Recovery

The brace is added only after the first uncertainty has forced a concrete change. The brace becomes a record of the recovery: it may stabilize a tilted top beam, lock an asymmetrical frame, or compensate for a shortened span.

## Uncertainty Ladder

### Layer 1: Material Uncertainty

Planned step:

The robot should place a 45 cm top beam across two posts spaced about 38 cm apart.

What goes wrong:

The top beam is slightly warped. When the robot lowers it, the left end touches the left post, but the right end sits 1-2 cm above the right post. The robot can keep holding the beam, but it cannot know whether the gap is acceptable, whether the beam should be rotated, or whether the post should move.

What the agent detects:

- The expected "both ends seated" condition has failed.
- The robot is in a holding state and should not release.
- The missing information is material judgement: fit, acceptable deformation, and desired aesthetic.

What the agent says:

> "Pause there. The left end is seated, but the right end is floating above the post. Please check the beam by hand: is it warped enough that we should rotate it, or should we move the right post inward?"

What each actor does:

- Robot: holds the top beam still at the current height.
- Human: presses lightly on the right end, rotates the beam 180 degrees, and checks whether the gap improves.
- Agent: compares two concrete recovery paths and names their consequences.

Recovery paths:

1. **Rotate the beam.** If rotating reduces the gap, the robot places it with the bend facing upward. The structure remains symmetric, but the top beam visibly arches.
2. **Move the post inward.** If the gap remains, the human moves the right post 2-3 cm inward while the robot keeps the beam lifted. The frame becomes narrower than planned.
3. **Use the beam as a brace.** If the beam is too warped for the top span, the robot places it diagonally later. Another straighter beam becomes the top.

What changes physically:

The frame is no longer a pure execution of the initial sketch. Its final geometry records a material negotiation: arched top, narrower span, or diagonal reuse.

Why this is symbiotic:

The robot contributes stable holding, the human contributes tactile material judgement, and the agent prevents a premature release while turning the observed gap into concrete design paths.

### Layer 2: Spatial Uncertainty

Planned step:

After the top beam is placed, the group needs to add a diagonal brace from the front-left base rail to the upper-right post.

What goes wrong:

The human points and says, "Put the brace there." The agent has possible gaze or pointing information, but there are two plausible targets: the brace could land on the outside face of the right post or the inside corner under the top beam. The robot can reach one target safely, but the other approach would pass close to the left post.

What the agent detects:

- The word "there" does not identify a unique contact point.
- The robot path depends on which contact point is meant.
- If the wrong target is chosen, the brace will either collide with the frame or change the design.

What the agent says:

> "I need to pin down 'there' before the robot moves. Do you mean the outside face of the right post, or the inside corner under the top beam? I can have the robot point to each one."

What each actor does:

- Robot: points first to the outside face, then to the inside corner, without carrying the brace.
- Human: confirms the intended contact by touching the target with a finger or temporary marker.
- Agent: converts the confirmation into a robot-safe instruction: "brace lower end on front-left base, upper end on inside corner under top beam."

Recovery paths:

1. **Robot places the brace.** If the confirmed contact is reachable, the robot places the brace while the human steadies the lower end.
2. **Human final placement.** If the robot can bring the brace near the site but cannot rotate into the final angle, the robot holds the brace 5 cm away and the human seats it.
3. **Move the assembly.** If neither target is reachable, the human slides the whole base 8-10 cm toward the robot, then the agent requests a fresh robot pose before trying again.

What changes physically:

The brace location becomes a negotiated spatial reference, not a guessed coordinate. If the structure is moved, the whole assembly shifts toward the robot side, making robot reach part of the final layout.

Why this is symbiotic:

Human pointing is fast but ambiguous. Robot pointing is slow but precise. The agent uses the robot as a spatial cursor so the human can disambiguate the build without needing coordinates.

### Layer 3: Role Uncertainty

Planned step:

The agent has just confirmed a brace placement. The robot is preparing to hold the brace while the human steadies the top beam.

What goes wrong:

While the robot is moving toward the brace, the human rotates the right base rail slightly to make the frame look more dynamic. This changes the geometry. The planned brace no longer reaches both intended contact points, and the robot is now acting on an outdated plan.

What the agent detects:

- The human has made an intentional design change.
- The robot's pending action is based on the previous geometry.
- The question is no longer "where does the brace go?" but "who leads the redesign now?"

What the agent says:

> "Stop. The base angle changed after I planned the brace. The robot's next move is now based on old geometry. Should we keep your rotated base and redesign the brace around it, or return the base to the earlier parallel layout?"

What each actor does:

- Robot: stops and retreats to a neutral holding pose.
- Human: states intent: "Keep the rotation. I like the frame leaning."
- Agent: accepts the human as design lead for this change and creates a new limited task for the robot.

Recovery paths:

1. **Preserve the human change.** The robot becomes a holder, not a placer. It holds the brace near the new diagonal while the human finds the exact contact points.
2. **Return to the prior plan.** The human realigns the base rails, and the robot performs the originally planned brace placement.
3. **Split the decision.** Keep the rotated base, but add a short wedge or spacer so the brace can still meet the top post.

What changes physically:

If the human change is preserved, the final frame becomes asymmetric. The brace may cross at a steeper angle. The structure now visibly contains a human-authored deviation, not only robot-executed precision.

Why this is symbiotic:

Role uncertainty is concrete because two actors are acting from different versions of the plan. The agent's job is to stop stale execution, make decision authority explicit, and give the robot a revised role that supports the human's changed intent.

## Symbiotic Collaboration Pattern

The scenario teaches a repeated loop:

1. Plan together.
2. Act through the actor best suited to the step.
3. Notice uncertainty.
4. Classify it.
5. Ask the actor with the right capability.
6. Recover through a revised design move.
7. Record the decision.

## Expected Workshop Outputs

Each group produces:

- A small timber frame.
- A named agent persona.
- A three-layer uncertainty story.
- A role map showing what the human, robot, and agent each contributed.
- A short reflection on how the final structure changed because of uncertainty.

## Why This Scenario Matters

This scenario makes uncertainty progressive. Participants first encounter the material as unreliable, then space as ambiguous, then collaboration itself as negotiable. The agent becomes valuable because it keeps the group moving without pretending that uncertainty can be removed.
