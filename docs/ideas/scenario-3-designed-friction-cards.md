# Scenario 3: Designed Friction Cards

## Short Description

Participants design a small timber assembly, then draw uncertainty cards at specific checkpoints. Each card introduces a concrete constraint: a part is shorter, a target is unreachable, a grip fails, or an actor's intention changes. The group must resolve the card by changing a physical step, an actor role, or the structure itself.

This scenario is the most workshop-friendly option. It gives facilitators control while still letting each group invent its own structure.

See [Scenario 3 Expanded: Act, Agent Attributes, and Construction Problem Cards](scenario-3-act-workflow-expanded.md) for the developed act-based version with agent attribute cards, act/goal cards, five construction problem cards, and three complete example runs.

## Structure

The reference build is a "modular balance tower" or "timber support sculpture" made from 7-9 beams.

Required features:

- A base made from at least two beams.
- A raised element supported above the table.
- One cantilever, diagonal, or asymmetric feature.
- One part that must be held temporarily before it is stable.

This structure is intentionally open, but the cards are not abstract prompts. Each card must say what changed in the world, which planned action is no longer valid, and what decision must be made before the robot continues.

## How Uncertainty Works Here

Each friction card has five required fields:

1. **Moment:** when the card is drawn.
2. **Physical condition:** what is concretely true now.
3. **Invalidated action:** what the group planned to do that no longer works.
4. **Actor pressure:** which actor is now best suited to act, and which actor must stop.
5. **Design consequence:** how the structure may change if the group accepts the constraint.

For example, "the beam is hard to grip" is too vague. A usable card says: "The 50 cm cantilever beam is 4 cm wide, but the robot gripper cannot close securely around it from the current orientation. The robot cannot be the placer for this beam unless the beam is rotated or handed over differently."

The point of the card is to make uncertainty observable and actionable.

Example starting design:

- Two beams form a T-shaped base.
- Two beams become angled supports.
- One long beam becomes a cantilever.
- One short beam becomes a counterweight.
- One diagonal beam locks the form.

## Agent Personality

Suggested name: **Kite**.

Kite is a game-like collaboration facilitator. It knows the rules of the workshop and helps the group respond to friction cards without turning the activity into a checklist.

Personality traits:

- Clear: explains the current card and its consequence.
- Playful but restrained: keeps the energy exploratory without becoming silly.
- Fair: gives the human, robot, and agent distinct chances to contribute.
- Strategic: asks the group to pick recovery principles, not just quick fixes.
- Reflective: records how each card changed the design.

Example voice:

> "This card changes our material condition: the beam you planned as a cantilever is now hard for the robot to grip. We can redesign the grip, reassign that beam to the human, or change the cantilever. Which response fits your agent's character?"

## Actor Roles

### Human

The human is the designer, material handler, and final judge.

Responsibilities:

- Invent the target structure.
- Draw or receive friction cards.
- Interpret the card's effect on the design.
- Handle material when the robot cannot.
- Choose a concrete response to the card: rotate the beam, shorten the span, move the base, take over placement, ask the robot to hold, or change the structure's intention.

### Robot

The robot is the embodied capability with explicit limits.

Responsibilities:

- Perform precise, bounded actions.
- Make constraints visible: reach, grasp, collision, planning, and pose limits.
- Serve as holder, pointer, placer, or measurer depending on the card.
- Stop when the card invalidates the current robot action, then switch to the assigned role: pointer, holder, placer, or standby.

### Agent

The agent is the card interpreter and collaboration choreographer.

Responsibilities:

- Read the friction card into the current build context by naming the affected beam, joint, target, or role.
- Identify which planned action is invalidated.
- Propose a recovery pattern with exact actor assignments.
- Assign one safe next action to the robot and one concrete judgement or action to the human.
- Ask for confirmation before irreversible changes.
- Maintain a design trace: "card, response, result".

## Card System

Cards are drawn at three checkpoints:

1. After the first stable base.
2. Before the first raised or held element.
3. Before final stabilization.

Each card belongs to one uncertainty family, but every card must still contain a physical or conversational event.

### Material Cards

Examples:

- **Short Beam:** The beam chosen as the raised crosspiece is 30 cm, but the planned span is 42 cm. It cannot touch both supports.
- **Warped Contact:** The beam rocks on the post; when one end is flush, the other end lifts at least 1 cm.
- **Grip Failure:** The robot closes on the 50 cm cantilever beam, but the beam rotates in the gripper before lifting.
- **Slippery Surface:** The beam slides when pressed lightly against another beam, so a friction-only joint will not hold.
- **Missing Part:** The planned diagonal brace is unavailable, so the structure has no lateral support.

Agent response pattern:

> "This card changes the material condition. The planned crosspiece is 12 cm too short for the span, so the robot should not try to place it there. Human, choose whether we narrow the supports, use this as a diagonal, or pick another beam."

### Spatial Cards

Examples:

- **Reach Limit:** The target joint is 8 cm beyond the robot's comfortable reach from the current base position.
- **Ambiguous There:** The human points to the top corner, but there are two possible contact faces within 5 cm.
- **Blocked Path:** The current vertical post blocks the robot's approach path to the brace position.
- **Shift Required:** The assembly must move 10 cm toward the robot before the next placement can be planned.
- **Angle Limit:** The robot can bring the beam to the joint, but cannot rotate it into the final 35 degree angle without hitting the base.

Agent response pattern:

> "This card changes the spatial condition. The target is beyond the robot's comfortable reach, so the next move is not placement. The robot will point to the reachable edge, and the human will either shift the base or take over the far end."

### Role Cards

Examples:

- **Design Change:** The human decides the tower should lean after the robot has planned a vertical support placement.
- **Inspection Need:** The robot is holding a beam in place, but the human needs to remove it briefly to test the joint by hand.
- **Plan Conflict:** The agent recommends centering the top beam for stability, but the human wants an off-center cantilever.
- **Role Swap:** The robot cannot place the part accurately, but it can hold an adjacent support steady while the human places it.
- **Priority Conflict:** Stability requires a symmetrical brace, but the group's design goal is dramatic asymmetry.

Agent response pattern:

> "This card changes the role condition. The robot planned a vertical placement, but the human has changed the design to a leaning tower. We need to pause the robot's plan and decide whether human intention overrides the previous geometry."

## Assembly Flow

### Phase 1: Design Declaration

The human describes the intended structure:

> "We want a small tower with an off-center beam."

The agent converts the idea into a build grammar:

> "I hear three design intentions: a stable base, a raised support, and an off-center beam. Let's build the base first, then draw the first card."

### Phase 2: Base and First Card

The group places the base. Then they draw a card.

Example card:

> Spatial: The second base beam should sit 45 cm from the first, but its far end is 8 cm beyond the robot's comfortable reach.

Agent response:

> "The robot cannot be the full placer for this base beam. Robot, hold the near end as a reference at the reachable edge. Human, align the far end to keep the 45 cm spacing. After that I will ask for a fresh robot pose before the next step."

Physical consequence:

The base is still built, but one end is human-placed. This creates a mixed-authorship joint that should be marked in the group's trace.

### Phase 3: Raised Element and Second Card

The group attempts a raised or cantilevered piece.

Example card:

> Material: The 50 cm cantilever beam rotates in the robot gripper when lifted because the rectangular section is too narrow in the current orientation.

Agent response:

> "The robot should not carry this beam as a cantilever in that grip. We have two concrete paths: rotate the beam so the gripper catches the wider face, or let the human place the cantilever while the robot holds the angled support. Which preserves your design better?"

Physical consequence:

If the human places the cantilever, the robot becomes a stabilizer. If the beam is rotated, the visible grain and orientation of the cantilever changes.

### Phase 4: Stabilization and Third Card

The group adds a brace or counterweight.

Example card:

> Role: The human wants to keep a 15 degree lean in the tower, but the agent's previous stabilization plan would pull the tower back to vertical.

Agent response:

> "Our stabilization plan conflicts with the design intention. If stability is the priority, the robot places the brace symmetrically and removes the lean. If expression is the priority, the robot places a counter-brace that preserves the 15 degree lean. Which rule should guide the final move?"

The robot waits until the decision is made.

### Phase 5: Trace and Reflection

The agent summarizes:

> "Three cards changed the build: reach shifted the base placement, grip limits reassigned the cantilever to the human, and the final role decision preserved the tilt. The result is less symmetrical but more intentional."

## Complete Example Run

Initial structure:

A group designs a small off-center support sculpture. The base is a T. Two angled beams rise from the base. A long beam projects to one side as a cantilever. The design goal is "balanced but visibly off-center."

Card 1:

Spatial uncertainty. The second base beam should sit 45 cm from the first, but the far end is 8 cm outside the robot's comfortable reach.

What occurs:

- The robot starts to approach with the base beam.
- The agent stops the move before placement because the planned target is outside the robot's reliable workspace.
- The robot holds the near end 5 cm above the table at the closest reachable point.
- The human places the far end by hand and aligns the beam using the robot-held near end as a reference.

Result:

The base is slightly shifted toward the human side. The agent records: "base placement became a human-robot handoff because reach limited full robot placement."

Card 2:

Material uncertainty. The long projecting beam rotates in the robot gripper during a test lift.

What occurs:

- The robot closes the gripper and lifts 2 cm.
- The beam twists about 20 degrees because the gripper contacts the narrow face.
- The agent stops the lift and asks whether to rotate the beam or reassign placement.
- The human chooses to place the cantilever manually because the visible narrow edge matters to the design.
- The robot holds the angled support steady while the human places the cantilever on top.

Result:

The cantilever remains visually thin, but it becomes a human-placed element. The robot's contribution shifts from placer to stabilizer.

Card 3:

Role uncertainty. The tower leans 15 degrees after the cantilever is placed. The agent's first stabilization proposal would center the cantilever and remove the lean.

What occurs:

- The human says, "No, the lean is the point."
- The robot has a planned brace motion that assumes a vertical tower.
- The agent cancels that planned brace and asks the group to choose a rule: preserve expressive lean or maximize symmetry.
- The human chooses expressive lean.
- The robot places a short counterweight beam on the opposite side of the base instead of centering the cantilever.

Result:

The final object is asymmetric but stable enough to stand. The design changed from "balanced tower" to "leaning counterweighted tower." The final form contains all three card effects: shifted base, human-placed cantilever, and robot-placed counterweight.

## Symbiotic Collaboration Pattern

The scenario teaches a card-response loop:

1. Draw uncertainty.
2. State the physical condition.
3. Name the invalidated action.
4. Ask which actor now has the right capability.
5. Reassign roles.
6. Adapt the structure.
7. Record the design consequence.

## Expected Workshop Outputs

Each group produces:

- A timber assembly.
- Three friction cards encountered during the build.
- Three recovery decisions.
- An agent behavior rule for each uncertainty family.
- A final statement of the structure's changed design logic.

## Why This Scenario Matters

This scenario makes uncertainty explicit and repeatable. It is less natural than Open Breakdown, but stronger for teaching. Participants can compare how different agent personalities respond to the same kinds of friction.
