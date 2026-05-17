# Voice Robot Agent Context

This glossary defines the project language for the Pipecat voice robot agent target architecture. It describes concepts and ownership, not transient implementation details.

## Language

### Voice Runtime

**Voice Runtime**:
The reusable runtime that turns realtime browser audio into an **Agent Turn** and assistant speech.

**Runtime Profile**:
App configuration that selects voice, agent, robot, and metrics adapters without constructing them.

**Voice Command**:
A wake-gated spoken command, normally beginning with "Mave".

**Agent Turn**:
One backend response to the latest user command, exposed to Voice Runtime as assistant text chunks.

**Voice Metrics**:
Semantic timing for wake, speech capture, STT, agent response, TTS first audio, and TTS completion.

**Voice Modulation**:
Provider-agnostic post-TTS audio shaping applied inside **Voice Runtime** before speech reaches transport output.

**Voice Mod Lab**:
The local web tuning app for generating TTS reference recordings, previewing **Voice Modulation**, and saving **Voice Modulation Presets**.

**Agent Persona Lab**:
The local web tuning app for tuning **Persona Prompt Parts**, the Gemini Live **TTS Voice Default**, and **Voice Modulation** for one **Runtime Profile**.

**Voice Modulation Preset**:
A saved set of validated audio-effect settings used by **Voice Modulation** for one **Runtime Profile**.

**Voice Modulation Default**:
The committed **Runtime Profile** voice-modulation setting that controls startup behavior before local lab overrides are applied.

**TTS Voice Default**:
The committed **Runtime Profile** TTS voice choice used by the live speech renderer after restart.

**Voice Runtime Assembly**:
The processor-ordering interface for transport input, optional Voice Command audio gate, STT, optional Voice Command transcript adapter, user aggregation, Agent Turn, TTS, transport output, and assistant aggregation.

### Agent Control

**Agent Control Module**:
The target module for API-key-backed LangChain intent handling and Agent Orchestration.

**Workshop Agent Location Contrast**:
The workshop research contrast between a **Robot-Inhabiting Agent** and a **Separate Floating AR Avatar**, where embodiment, voice source, visual position, visual appearance, persona, and authority cues are intentionally tunable by participants.

**Robot-Inhabiting Agent**:
A workshop agent state where participants design the agent as perceived through, or as part of, the robot body. The agent may use participant-facing body language such as "my hand" while operational instructions still ground planning in the UR10, MoveIt, and TCP/end-effector terms.

**Separate Floating AR Avatar**:
A workshop agent state where participants design the agent as a distinct AR participant beside the robot rather than as the robot.

**Agent Location Fit**:
How well a participant-tuned agent location and embodiment configuration fits a specific construction uncertainty, including the tradeoffs it creates for authority, clarity, trust, responsibility, and collaboration.

**Post-Run Evaluation Questions**:
The stable workshop questions participants answer after each construction run to compare Agent Location Fit across tuned agent states.

**Starter Agent Persona Card Deck**:
A workshop card deck of initial agent concepts that participants adapt into paired Robot-Inhabiting Agent and Separate Floating AR Avatar states.

**MAVE Starter Persona**:
A robot-oriented starter persona where the agent feels like a self-directed machine-body collaborator, with expressive movement cues and a little independent judgment, while still respecting robot-control limits and human material judgment.

**Kibbitz Persona**:
The current agent character used across embodiment setups; workshop contrasts should first vary embodiment framing, not replace the character.

**Robot Nonverbal Cues**:
Small bounded robot movements that communicate turn-taking, attention, excitement, hesitation, refusal, or confirmation without changing the construction state.

**LangChain API Backend**:
The native LangChain chat-model backend accessed with provider API keys.

**Agent Orchestration**:
Dialogue and tool-loop control behind the Agent Turn seam; LangGraph may own this, but Pipecat remains the Voice Runtime owner.

**Robot Agent Prompt**:
The concise behavior prompt aligned with Agent Orchestration, robot tool feedback, and the robot tool contract.

**Persona Prompt Part**:
A versioned prompt part that tunes agent identity, embodiment, speech delivery, response style, or behavior examples without changing the robot tool contract.

**Agent Embodiment Setup**:
A selectable **Persona Template** setup for presenting the agent as either a **Separate Floating AR Avatar** or a **Robot-Inhabiting Agent**.

**Persona Template**:
A versioned source template for loading a coherent set of **Persona Prompt Parts** into the editable prompt files. A template is self-contained even when some files are unchanged from another template.

**Canonical Motion Examples**:
Required robot-behavior examples that keep **Agent Orchestration** aligned with current MoveIt tool workflows.

**Behavior Examples**:
Editable persona-tuning examples that show how the agent should interact with the system in selected user scenarios.

### Robot Control

**Robot Control Module**:
The target module for robot-side control concerns: Task Policy, Robot Call Validation, Robot Tool Adapter, and Robot Context.

**Robot Observation**:
A fresh robot-state read before movement, retries, relative commands, or safety-sensitive actions; last-known context is advisory only.

**Robot Context**:
Advisory recent robot observations, planning results, gripper state, execution results, and held objects proven by attach or attached-object verification evidence.

**Shared Geometry Model**:
A geometry-operation model, expressed as abstract primitives, transforms, geometric features, and constraints, exchanged between the agent and the ROS/Vizor environment for spatial manipulation and synchronization.

**Geometry World Model**:
The paired **Physical Geometry Model** and **Hologram Geometry Model** used by the agent to reason about element identity, roles, and placement intent.

**Physical Geometry Model**:
The **Shared Geometry Model** view of physical elements that exist in the MoveIt planning scene and can be picked or placed by the robot.

**Physical Model Pose Update**:
A deterministic bookkeeping mutation of the **Physical Geometry Model** from named planning-scene object pose evidence after verified release/place proof or operator sync intent, not from raw robot TCP pose alone.

**Physical Model Sync**:
An explicit operator-triggered update that reconciles the **Physical Geometry Model** from current named MoveIt/RViz object state.

**Physical Model Update Helper**:
The deterministic Robot Control helper that mutates the **Physical Geometry Model** from named MoveIt/RViz object pose evidence after verified execution or operator sync intent.

**Dynamic Role**:
The semantic structural role of a dynamic object, expressed as a relation such as `supporting_column` or `beam_supported_by(dynamic_2,dynamic_3)`, or `unassigned` when no role is known.

**Dynamic Role Update Tool**:
The `geometry_update_dynamic_role` LangGraph tool that mutates a dynamic object's **Dynamic Role** only when Agent Orchestration has clear structural/contact understanding or human confirmation.

**Dynamic Role Payload**:
A structured role object for `unassigned`, `supporting_column`, or `beam_supported_by` relations; not free prose.

**Physical Model Update Reason**:
The constrained reason enum for a **Physical Model Pose Update**: `verified_pick_place_release`, `verified_place_release`, or `operator_sync`.

**Full Object Pose Evidence**:
Named dynamic-object evidence containing both position and orientation quaternion from MoveIt/RViz or verified placed-object proof.

**Pose-Derived Geometry**:
The **Physical Geometry Model** fields whose values follow from an element object pose, including body pose, axis endpoints, and pose-dependent feature centers.

**Dynamic Object Local X Axis**:
The beam centerline direction derived from a dynamic object's orientation quaternion and used to recompute axis endpoints from pose and length.

**Calibrated Workspace Coordinates**:
The shared numeric coordinate space where Grasshopper-authored element positions match the RViz/MoveIt scene positions.

**Hologram Geometry Model**:
The **Shared Geometry Model** view of user-positioned AR target elements that express desired object poses without adding collision geometry to MoveIt.

**Hologram Target Pose**:
The desired beam/object pose from the **Hologram Geometry Model**, not a robot TCP pose.

**Beam Orientation Grasp Coverage**:
The requirement that supported dynamic-beam manipulation can solve grasps for both horizontal beams and vertical beams when the live MoveIt planning scene exposes the object geometry.

**Geometry World Context**:
A compact Agent Orchestration instruction block rendered from the **Geometry World Model** for the current turn.

**Geometry-Grounded Pick-Place**:
A single compound pick-place task that moves one physical `dynamic_*` object to its matching **Hologram Target Pose**.

**Hold Compound Goal**:
The `requirements.goal="hold"` **Compound Task Plan** for natural requests such as "pick up" or "grab and lift". It grasps and attaches the object, then includes an agent-specified, bounded post-grasp lift so the object is visibly held; default lift is `0.10` m and v1 accepts `0.03`-`0.20` m. It does not relocate the object to a hologram target pose.

**Canonical Dynamic Name**:
The unpadded `dynamic_1`-style object name used to pair MoveIt scene objects with shared geometry bodies.

**Task Policy Layer**:
A deterministic pre-tool layer for obvious robot-step preconditions before Robot Call Validation and MoveIt; v1 covers fresh pose before motion, no blind execute, and basic gripper/attach ordering.

**Task Policy Decision**:
The structured allow/block result from the Task Policy Layer, with correction text and a suggested next tool when a step is blocked.

**Robot Call Validation**:
Lightweight local validation for allowed MoveIt tool names, UR10 robot name, argument shape, bounded task parameters such as v1 hold lift distance from `0.03` to `0.20` m, target bounds, timeouts, canonical-to-legacy tool names, executable plan names, and clearer error text; it is not a task policy layer and is not the source of movement safety.

**Robot Tool Adapter**:
The Agent/Robot Control seam that exposes and executes robot tools while routing movement through MoveIt workflows and normalizing MCP timeouts/exceptions.

**Robot Job Blackboard**:
The shared typed job/event surface for long-running robot action execution. Agent Control writes queued robot jobs; Robot Control workers write started, completed, and failed events.

**Robot Job Worker**:
A deterministic Robot Control worker that validates and executes the exact queued MoveIt tool call. It does not invent new tool calls, repair arguments, or make LLM decisions.

**Executable Plan**:
A successful MoveIt planning result with `ok=true`, `feedback.can_execute=true`, and a valid returned `raw.plan_name` that can be executed through a MoveIt execution workflow.

**Task Solution**:
A successful task-level MoveIt MCP planning result with `ok=true`, `feedback.can_execute=true`, and a valid returned `raw.task_solution_id` for ordered pick/place/compound stages, bound to the **Scene Snapshot Evidence** used at planning time. It is planning evidence, not physical execution evidence.

**Task Solution Cache**:
The MCP-owned immutable store keyed by `task_solution_id`, containing the solved task payload, execution contract, scene snapshot evidence, preview evidence, creation time, and expiry evidence used later by approved task execution.

**Verified Real Robot Execution**:
The host-side actuation boundary that executes cached MoveIt plans on the physical UR10 and Robotiq path after explicit execution intent.

**Verified Execution Server**:
The repo-local service that exposes the **Verified Real Robot Execution** HTTP boundary. It may use UR RTDE Receive for readiness, TCP pose, joint state, and completion evidence directly from the UR controller, but production motion uses URScript over the robot script socket and gripper control uses the direct Robotiq socket.

**Narrow Verified Execution Migration**:
The workshop migration rule that brings only the UR10/Robotiq verified execution service code into `server/verified_execution_server`, not the broader legacy multi-robot `core` and `devices` framework.

**Simulation-Only Robot Execution**:
Legacy wording for a runtime where physical execution is unavailable. The task executor still runs the digital/AR/RViz path and reports physical execution as unavailable rather than using a separate user-visible execution mode.

**Simulation-First Dual Execution**:
The robot execution mode where approved task execution is invoked as one task-level call, while AR/RViz and physical execution advance through the same ordered stage evidence when physical execution is connected. The implementation always runs the RViz/MoveIt and AR visualization path, attempts **Verified Real Robot Execution** when connected, and reports digital and physical result facts separately. The digital path is required execution feedback for AR, not optional debug output.

**Physical Execution Unavailable**:
The verified physical robot branch did not respond or is not connected. This does not block the digital/AR/RViz task path; it is reported as unavailable rather than as task failure.

**Physical Execution Failure**:
The verified physical robot branch responded but failed execution or verification. If the digital/AR/RViz path succeeded, user-facing speech should say that execution completed in AR/RViz but physical execution failed.

**Unified Task Execution Tool**:
The single model-visible task execution tool, `moveit_execute_task`, that executes an approved **Task Solution** while older execution tools remain internal compatibility paths.

**Monolithic Task Execution Call**:
A single approved task-level execution call that preserves a clean agent-facing contract while internally executing and reporting ordered digital/AR, physical motion, gripper, attach/release, and verification stages. It must not be an opaque timeout-prone wrapper.

**Stage-Synchronized Dual Execution**:
The rule that one execution-contract stage identity ties together digital/AR/RViz proof and physical proof. A motion stage's AR preview, MoveIt plan evidence, and verified physical execution all refer to the same approved stage and plan evidence.

**Physical Alignment Probe**:
A read-only UR RTDE Receive observation used to compare the connected UR controller's actual joints or TCP pose against expected execution state. It is used only when the robot responds; it must not become a requirement for the digital/AR/RViz path when the physical robot is unavailable.

**Workshop Monorepo**:
The single repository students clone for the workshop runtime, including the voice agent, operator dashboard, Vizor/MoveIt Docker stack wiring, MCP services, and verified execution server.

**Workshop Runtime Services**:
The repo-local runtime service packages under `server/`, including the voice agent packages, operator dashboard, MoveIt MCP, Vizor MCP, and verified execution server.

**Operator Dashboard**:
The `server/operator_dashboard` service that starts and monitors the workshop runtime services and the **Canonical Development Compose** stack from a local browser UI.

**Workshop Operator Config**:
The repo-root `configs/` operator-facing configuration for dashboard service commands, robot IPs, and workshop startup defaults. Committed config and docs must use repo-relative paths or portable defaults, not machine-specific absolute paths.

**Workshop MCP Services**:
The repo-local `server/moveit_mcp` and `server/vizor_mcp` packages. They keep their package names so the Docker stack can continue to launch `python -m moveit_mcp` and `python -m vizor_mcp`.

**Workshop MCP Image**:
The Docker image built from the **Workshop Monorepo** that packages only the workshop MCP service code and its minimal runtime dependencies. Compose keeps both prebuilt image tags and repo-local build contexts so instructors can distribute images while developers can rebuild from source.

**Legacy Multi-Actor Material**:
Older Multi-Actor assignment code, Gradio agent surfaces, Mongo sensor workflow, broad multi-robot framework code, and study artifacts that are not part of the current Pipecat workshop runtime.

**Workshop Dashboard Launcher**:
The root-level Windows `.cmd` entrypoint that verifies `uv`, performs first-run server dependency sync when needed, starts only the **Operator Dashboard**, and keeps startup failures visible for workshop participants. The dashboard URL with its token is printed by the Python dashboard runner in the `.cmd` window, and the runner may auto-open that URL in the browser. The `.cmd` keeps the dashboard process in the foreground so logs and errors remain visible. Service lifecycle remains visible inside the dashboard UI. The launcher does not create local config files.

**Canonical Development Compose**:
The repo-local versioned Docker Compose configuration in the **Workshop Monorepo** that owns the Vizor/MoveIt Docker stack. It owns development image tags, service wiring, and MTC enablement; local operator configuration is limited to machine-specific environment and secret overrides, which must not be committed.

**Verified Task Plan Execution Bridge**:
The internal Robot Control bridge used by a **Monolithic Task Execution Call** to realize stage-by-stage task execution. It consumes a recent approved **Task Solution** with a supported `execution_contract`, retries task motion stages when needed, executes returned plan names through **Verified Real Robot Execution**, interleaves verified gripper actions with MCP attach/release tools, and verifies attachment or release before success.

**Task-Level Pick**:
A MoveIt MCP pick workflow that plans observe, approach, gripper, attach, lift, and attachment-verification stages as one **Task Solution**.

**Task-Level Place**:
A MoveIt MCP place workflow that plans object placement stages as one **Task Solution** and still requires execution plus release or placed-object evidence before a success claim.

**Task-Level Manipulation Planner**:
The one model-visible planner, `moveit_plan_manipulation_task`, for manipulation requests such as hold, pick-place, move-and-release, place, and release. It accepts the desired object, goal, and target pose requirements, then returns a **Task Solution** only when the selected backend can produce previewable, executable, proof-backed stages.

**Manipulation Goal**:
The `requirements.goal` value for `moveit_plan_manipulation_task`. Supported goals are `hold`, `place`, `release`, `move_and_release`, and `pick_place`; natural "pick up" maps to `hold`, not to a separate `pick` goal.

**Staged MoveIt Manipulation Backend**:
The explicit v1 backend for the **Task-Level Manipulation Planner**. It composes existing MoveIt/Vizor free-motion, Cartesian-motion, gripper, attach, release, and verification steps into one **Task Solution**. It is not an MTC backend and is not a silent fallback.

**Recovery-Oriented Manipulation**:
The staged backend design goal that planning should try multiple valid grasp, approach, distance, and orientation candidates before failing, and should return clear failed-stage feedback when no candidate works.

**Manipulation Recovery Boundary**:
The rule that broad candidate search belongs before physical execution. After execution starts, retries are limited to the same approved task stage shape and must not invent a new semantic task after gripper close, attach, or object motion without a new approved **Task Solution**.

**Structured Candidate Search**:
The staged manipulation planning strategy where "more attempts" means trying different bounded grasp faces, approach distances, standoff distances, beam-orientation strategies, and motion planners, not blindly repeating the same failed request. Candidate search has a fixed v1 budget of up to 8 grasp candidates and up to 4 motion attempts per candidate, and reports all tried candidates when planning fails.

**Hybrid Manipulation Stage Planning**:
The staged manipulation motion policy where far approach to pick pre-grasp, far approach to place pre-pose, and held-object travel use free-motion planning, while contact-sensitive final approach, lift, descent, and extraction use Cartesian planning. Preview playback speed is separate from planner choice.

**Beam Grasp Strategy**:
The staged manipulation grasp policy for construction beams aligned to X or Y. Horizontal beams use the robot-frame top face as the normal grasp and do not automatically try side grasps unless the user explicitly asks for or approves side-grasp recovery. Vertical beams use side grasps, not the robot-frame top cap; the preferred "outer side" is a ranked side-face choice based on planning-scene clearance and facing away from the assembly center or nearest neighboring beam, while still allowing all valid side faces within the candidate budget.

**Manipulation Planning Observation**:
The fresh backend-side object and robot-state evidence fetched by the **Task-Level Manipulation Planner** before planning. Agent Orchestration may observe first for dialogue or disambiguation, but the planner must not rely on stale model-supplied object context as authoritative planning input.

**Manipulation Failure Feedback**:
Structured failed-planning or failed-execution feedback containing `failed_stage`, `failure_code`, `tried_candidates`, `what_was_proven`, `what_is_uncertain`, `suggested_next_action`, and a concise human-facing message.

**Manipulation Recovery Question**:
A plain user-facing question asked by Agent Orchestration after the staged backend exhausts clearly allowed automatic candidates and the next recovery option changes the intended grasp, risk, object setup, or requires human judgment.

**Manipulation Plan Success**:
A manipulation plan succeeds only when required motion stages are planned with non-empty trajectories, `AgentPath` preview evidence exists, the execution contract is complete, scene snapshot evidence exists, and approval evidence is prepared. Advisory integration facts such as AR subscriber absence or optional physical-model pose-update evidence do not block planning success.

**Release Intent**:
Agent Orchestration's semantic interpretation of requests such as "drop it" or "let go" as releasing the currently held or attached object through a verified release workflow. It is not an uncontrolled physical drop or raw gripper command.

**Release Compound Goal**:
The `requirements.goal="release"` **Compound Task Plan** for releasing the currently held or attached object without first moving it to a new target pose. A move followed by release uses `move_and_release`. Plain release has no robot motion trajectory, so it reports no-motion preview evidence instead of publishing `/UR10/request/planned_path`.

**Compound Task Plan**:
An MTC-only MoveIt MCP workflow planned by `moveit_plan_compound_task` from hard `requirements` and optional `preferences`. `requirements.goal` is limited to `hold`, `release`, `move_and_release`, or `pick_place`; there is no separate `pick` goal. `requirements.goal`, `requirements.object_name`, and goal-specific requirements such as `lift_distance_m` define the public task; `preferences` may bias grasp selection but do not command the grasp algorithm. Unsupported hints such as `slide`, `push`, raw code, or raw waypoints fail at planning with no task solution id.

**Execution Contract**:
A proof-backed ordered contract inside a **Task Solution**. Each step names a supported handler, source stage, required proof, object, and scene snapshot context. Planning returns no `task_solution_id` when a solved MTC stage lacks a typed handler or proof requirement. `moveit_execute_task` rejects unknown handlers, unsupported task kinds, raw waypoint-only recipes, stale approval, materially stale scene snapshots, and missing proof fields.

**MTC Failure Code**:
A stable machine-readable reason for failed compound planning or execution readiness, such as `object_not_found`, `not_holding_object`, `unsupported_grasp_orientation`, `no_ik_solution`, `collision`, `preview_publish_failed`, or `stale_scene`, paired with `retryable`, `correction`, and `suggested_next_tool` when applicable.

**MTC Backend**:
An optional MoveIt Task Constructor implementation backend for task-level tools. The default pick task backend remains emulated. Compound task planning requires explicit `backend="mtc"`. Configured MTC success requires solved MTC stage evidence and a task solution; failure returns `ok=false`, `failed_stage`, and `blocker` without a silent emulated fallback. Current ROS service endpoints are `/vizor_mtc/plan_pick_task` and `/vizor_mtc/plan_compound_task`.

**MTC Task Preview**:
The operator-visible RViz/Vizor preview of a solved MTC task solution, published by the MTC backend on `/solution` for the RViz Motion Planning Tasks display. It is planning/preview evidence, not physical execution proof.

**AR Planned Trajectory Preview**:
The Vizor AR planned-motion preview carried on `/UR10/request/planned_path` as `vizor_package/PlannedTrajectory`, with `name`, `platform_name`, and `trajectory_msgs/JointTrajectory joint_trajectory`. The public AR path name for agent-planned manipulation is `AgentPath`. A composed trajectory for the whole motion is preferred; explicit staged preview is acceptable when a composed trajectory cannot be exported, using ordered stage names such as `AgentPath:01_approach`. The system must not synthesize a fake composed trajectory. The publisher authority is `/vizor_robot_control`; the MTC backend, MCP, and Pipecat must not become competing AR preview publishers. Publication failure for a motion-bearing task is a planning blocker, while subscriber absence is advisory integration evidence.

**AR AgentPath Execution**:
The AR execution surface where a human presses the AR execute button and Vizor publishes `std_msgs/String` with payload `AgentPath` on `/UR10/command/execute`. For manipulation tasks, `AgentPath` means the whole approved goal, not one internal stage. Stop and gripper buttons remain operator controls and do not replace task-solution proof.

**AR AgentPath Stop**:
The AR stop surface where Vizor publishes `std_msgs/String` with payload `AgentPath` on `/UR10/command/stop`. It cancels the active AgentPath task, invalidates the cached task execution, and requires Robot Control to re-observe the robot, gripper, attached object, and planning scene before replanning.

**AR Manual Gripper Control**:
The AR gripper debug surface where Vizor publishes `std_msgs/Bool` on `/Robot/gripper`, with `true` and `false` representing operator gripper commands. It is for debugging, not normal manipulation execution or HITL task recovery, and it is not attachment proof or release proof.

**Partial Pick Diagnostic**:
A failed legacy pick planning result where only a preposition or earlier segment solved. It is diagnostic evidence, not an executable pick.

**Execution Approval Payload**:
Structured approval evidence bound to the exact plan or **Task Solution**, source tool, object, expected movement, scene snapshot, approval source, approval turn, and approval time. Task-solution approval expires after 60 seconds or the current spoken approval turn, whichever is stricter. Approval may come from spoken confirmation or an explicit AR execute action bound to the current `AgentPath`.

**Scene Snapshot Evidence**:
Compact evidence that binds a planning result to the grounded scene object, planning frame, pose age, `scene_snapshot_id`, and a normalized relevant-scene hash. The hash includes robot joint state, attached objects, target object pose and shape, relevant collision objects, and planning frame; it excludes unrelated metadata, preview publication status, and subscriber counts.

**Material Scene Change**:
A planning-scene difference that invalidates a **Task Solution**: robot state, attached objects, target object pose or shape, or relevant collision objects changed. Unrelated metadata does not invalidate the task solution.

**MoveIt Safety Boundary**:
The accepted movement-safety boundary; robot movement safety is delegated to MoveIt planning/execution and the robot simulation stack.

### Observability

**Process Trace**:
The reusable cross-cutting trace of a voice robot run, recorded as correlated local spans and events across Voice Runtime, Agent Control, Robot Control, and MCP tool execution.

**Trace Session**:
One bot runtime session, used to correlate all turns and background lifecycle events in a single run.

**Trace Turn**:
One spoken user command inside a Trace Session, normally rooted at wake or first user speech and ending after the assistant response/TTS path completes.

**Trace Span**:
A timed Process Trace record with a name, parent span, status, timestamps, duration, and structured attributes.

**Trace Event**:
An instant Process Trace record for a semantic event such as wake detection, policy block, validation result, Robot Context update, or lifecycle event.

### Testing And Replay

**Replay Artifact**:
A compact local artifact recording tool order, typed tool outputs, policy decisions, validation results, approvals, execution results, verification results, and terminal job events for review and replay.

## Relationships

- **Voice Runtime Assembly** contains exactly one **Agent Turn** processor in the voice pipeline.
- **Voice Modulation** belongs to **Voice Runtime** and runs after TTS, before transport output.
- **Agent Persona Lab** is the broader tuning surface for persona prompt parts, Gemini Live voice choice, and **Voice Modulation**.
- **Voice Mod Lab** may call TTS providers for reference recordings, but it saves local **Voice Modulation Presets** as tuning overrides.
- **Runtime Profile** may carry a committed **Voice Modulation Default**; local **Voice Modulation Presets** override it when present.
- **Voice Mod Lab** may update the committed **TTS Voice Default** for Gemini Live profiles, but the running bot uses it only after restart.
- **Agent Persona Lab** writes only narrow **Runtime Profile** voice fields and allowlisted **Persona Prompt Parts**; it is not a general configuration editor.
- **Agent Persona Lab** may preview unsaved editor text, but live **Agent Orchestration** uses saved prompt files loaded on bot start.
- **Agent Orchestration** happens behind **Agent Turn** and does not reorder **Voice Runtime Assembly**.
- **Agent Control Module** satisfies the **Agent Turn** backend seam and may use **Robot Control Module**.
- **Persona Prompt Parts** are versioned source prompts; saving them through a local lab UI changes future bot starts, not the already-running prompt constants.
- **Agent Embodiment Setup** switches through full **Persona Template** folders, not one-off prompt-part patches.
- **Robot-Inhabiting Agent** and **Separate Floating AR Avatar** presets may share the same **Kibbitz Persona** so the workshop contrast isolates embodiment framing.
- **Robot-Inhabiting Agent** prompts may support embodied spoken wording, but robot tool use remains grounded in UR10, MoveIt, and TCP/end-effector terminology.
- **Robot-Inhabiting Agent** prompt templates should align embodiment setup, speech delivery, and behavior examples while preserving the shared character persona unless the workshop explicitly tests character changes.
- **Persona Template** loading copies versioned template content into the editable **Persona Prompt Parts**; it does not create an untracked runtime mode.
- **Canonical Motion Examples** stay separate from editable **Behavior Examples** so persona tuning cannot silently weaken required robot workflow examples.
- **Behavior Examples** are included after **Canonical Motion Examples** so required robot workflow examples keep priority.
- **Voice Runtime** must not own **Task Policy Layer**, **Robot Call Validation**, **Robot Tool Adapter**, or **Robot Context**.
- **Shared Geometry Model** may inform agent spatial reasoning, but the **MoveIt Safety Boundary** remains authoritative for planning and execution safety.
- The **Geometry World Model** consists of the **Physical Geometry Model** and **Hologram Geometry Model**.
- **Physical Geometry Model** and **Hologram Geometry Model** pair bodies by **Canonical Dynamic Name**.
- Padded names such as `dynamic_01` may be accepted at tool boundaries, but tool results and planning calls should use the **Canonical Dynamic Name**.
- **Geometry World Context** exposes semantic placement only as **Dynamic Role Payload** at `role`.
- **Dynamic Role** values are structural/contact semantics: `unassigned`, `supporting_column`, or `beam_supported_by`.
- View-dependent placement labels such as left/right are not **Dynamic Role** values; use `unassigned` until a structural/contact role is known.
- **Dynamic Role Payload** lives in the **Physical Geometry Model**, not the **Hologram Geometry Model**.
- `geometry_update_dynamic_role` accepts a **Dynamic Role Payload**, not a free-text role string.
- **Dynamic Role Update Tool** is local to LangGraph/Agent Control, not an MCP tool.
- `geometry_update_dynamic_role` rewrites only **Dynamic Role Payload** plus operation history.
- `geometry_update_dynamic_role` validates referenced canonical dynamic names against `physical_model.json`.
- Agent Orchestration must ask the human when structural/contact role semantics are uncertain.
- **Dynamic Role Update Tool** returns structured feedback with `ok`, `object_name`, `role`, and `physical_model_updated` on success, or `ok=false`, `error`, `correction`, and `retryable` on failure.
- `geometry_update_dynamic_role` appends compact `dynamic_role_update` operation history.
- **Dynamic Role Update Tool** stays model-visible alongside manipulation tools, but it records semantic assembly meaning only; it is not robot motion, execution approval, attachment proof, release proof, or physical placement proof.
- A **Physical Model Pose Update** uses **Full Object Pose Evidence** as the primary object-pose evidence.
- UR RTDE TCP pose may support **Physical Model Pose Update** only as execution evidence or when deriving an object pose from a known attached-object grasp transform.
- A **Physical Model Pose Update** is deterministic bookkeeping after verified release/place proof or an explicit **Physical Model Sync**; passive object observations remain read-only.
- **Physical Model Update Helper** accepts only a valid **Physical Model Update Reason**.
- **Physical Model Update Helper** updates exactly one **Canonical Dynamic Name** per call.
- Automatic verified task execution updates the **Physical Geometry Model** only from full object pose carried by verified release/place proof; fresh MoveIt/RViz object context is allowed only for explicit `operator_sync`.
- Verified task execution may call the **Physical Model Update Helper** after release or placed-object proof only when proof includes **Full Object Pose Evidence**; Agent Orchestration does not need a separate update tool call.
- Verified task execution reports robot execution proof and **Physical Model Pose Update** outcome as separate result facts.
- A **Physical Model Pose Update** rewrites only **Pose-Derived Geometry** plus operation history; semantic identity fields and assembly structure stay stable unless a later operation explicitly changes them.
- **Physical Model Pose Update** derives axis endpoints from the object pose center, `solid.dimensions.x`, and the **Dynamic Object Local X Axis**.
- **Physical Model Pose Update** assumes **Calibrated Workspace Coordinates** between Grasshopper and RViz/MoveIt rather than applying a transform.
- **Physical Model Pose Update** does not infer **Dynamic Role** or assembly relations from pose alone.
- **Physical Model Update Helper** fails closed with structured feedback and no file write when named object evidence, valid geometry, allowed reason, or valid JSON is missing.
- Bounds centers and alignment axes are diagnostic only; they are not enough for a **Physical Model Pose Update** without a real orientation quaternion.
- **Physical Model Update Helper** writes `physical_model.json` atomically.
- **Physical Model Pose Update** is one-way bookkeeping from proved MoveIt/RViz dynamic-object pose to `physical_model.json`; it does not write back into MoveIt/RViz.
- **Geometry World Context** gives Agent Orchestration the paired elements, semantic labels, model names, and hologram target poses each turn.
- A **Hologram Target Pose** can become a MoveIt pick-place target pose only after the matching physical object is observed in the planning scene.
- **Geometry-Grounded Pick-Place** uses the MoveIt planning scene for the source object pose and the **Hologram Geometry Model** for the target object pose.
- Agent Orchestration gets hologram target poses through **Geometry World Context**, not by ad hoc JSON reads or extra target-pose tool calls.
- MTC compound planning uses the live MoveIt planning scene for current object geometry and **Geometry World Context** for any **Hologram Target Pose** selected by Agent Orchestration.
- Supported MTC beam grasping must satisfy **Beam Orientation Grasp Coverage** before the workflow is considered complete for construction beams.
- Agent Orchestration maps natural "pick up" requests to the **Hold Compound Goal**; it must not call or invent a separate `pick` compound goal.
- Agent Orchestration chooses the **Hold Compound Goal** lift distance through bounded `requirements.lift_distance_m`; prompt default is `0.10` m, and Robot Control validates the v1 `0.03`-`0.20` m bounds before planning.
- Missing or invalid **Hologram Target Pose** data blocks **Geometry-Grounded Pick-Place** with structured feedback; it must not fall back to the physical model or current object pose.
- **Geometry-Grounded Pick-Place** is planned through the **Task-Level Manipulation Planner**, not by exposing separate pick and place task planners to the model.
- **Task Policy Layer** runs before **Robot Call Validation**.
- **Robot Call Validation** may reject malformed tool calls, but it does not validate task-level intent and is not the source of movement safety.
- **Robot Tool Adapter** routes movement through the **MoveIt Safety Boundary** and normalizes MCP transport failures into structured robot feedback.
- **Robot Context** records held objects only from attach or attached-object verification evidence, and clears held state only from release proof.
- Before executing or continuing a manipulation task, Robot Control must verify held/attached state against current MCP/MoveIt evidence when that state matters; stale **Robot Context** alone is not enough.
- Agent Orchestration may infer **Release Intent** from natural language such as "drop it", but execution still requires the held-object context and verified release proof.
- **Release Intent** maps to the **Release Compound Goal** when no relocation target is requested, and to `move_and_release` when the user asks to move the held object before release.
- The **Release Compound Goal** requires fresh robot state plus **Robot Context** evidence that the named object is currently held or attached; it must not plan a release from object name alone.
- **Robot Job Blackboard** decouples slow robot action execution from the spoken **Agent Turn**.
- **Robot Job Worker** owns deterministic execution of queued robot jobs and writes terminal events back to the **Robot Job Blackboard**.
- An **Executable Plan** may be auto-executed only through a MoveIt execution workflow.
- **Simulation-Only Robot Execution** is selected by `robot_execution.simulation_only = true` and is the default mode for RViz/noVNC testing.
- **Canonical Development Compose** owns repeatable Vizor/MoveIt development stack wiring; local operator configuration must not hide MTC enablement or image-tag choices.
- A **Task Solution** is executed through `moveit_execute_task`, preserving the **Monolithic Task Execution Call** shape while internally producing stage-by-stage proof through the **Verified Task Plan Execution Bridge**. MCP-owned monolithic execution is an internal compatibility path only if it provides equivalent stage proof.
- The **Task-Level Manipulation Planner** may use the **Staged MoveIt Manipulation Backend** now and an **MTC Backend** later, but Agent Orchestration should still see `moveit_plan_manipulation_task` rather than competing pick, place, compound, and low-level motion tools.
- The **Verified Task Plan Execution Bridge** supports typed, proof-backed contracts for pick, place, hold, move-and-release, approach-hold-adjust-release, and pick-place task kinds.
- Verified task execution keeps the agent path semantic: task planner, explicit approval, then `moveit_execute_task`.
- **Task Solution Cache** owns immutable solved task payloads; approved execution reads the cached execution contract and bound scene evidence instead of trusting a model-restated contract.
- `moveit_execute_task` recomputes **Scene Snapshot Evidence** through Robot Control/MCP at execution time; Agent Orchestration never computes or supplies the scene hash.
- A live solved **MTC Backend** result must publish an **MTC Task Preview** when preview is part of the workflow; execution success still requires later attachment or release proof.
- Every motion-bearing manipulation task should produce or verify an **AR Planned Trajectory Preview** before execution through the `/vizor_robot_control` publisher authority. The primary AR path name is `AgentPath` and represents the whole goal; staged details may use ordered names such as `AgentPath:01_approach`, `AgentPath:02_pre_grasp`, and `AgentPath:03_lift`. Publication failure blocks planning, and lack of an AR subscriber is reported clearly but is not a v1 planning blocker.
- A plain **Release Compound Goal** is not motion-bearing and reports no-motion preview evidence rather than publishing an **AR Planned Trajectory Preview**.
- **Task Solution** execution requires the current planning scene to still match the bound **Scene Snapshot Evidence**; the normalized hash covers only planning-relevant scene facts, and **Material Scene Change** requires replanning.
- A **Partial Pick Diagnostic** must not be stored as an **Executable Plan** or **Task Solution**.
- Compound-task failures use **MTC Failure Code** values so Agent Orchestration can distinguish retry, correction, and HITL paths without parsing prose.
- A blocked **Task Policy Decision** is returned to **Agent Orchestration** as structured tool feedback, not as a movement-safety claim.
- A **Replay Artifact** preserves the task-solution workflow evidence needed to review observe, plan, approve, execute, verify, and summarize loops.
- **Process Trace** observes runtime behavior but does not own Voice Runtime, Agent Control, Robot Control, policy, validation, MCP execution, or robot safety behavior.
- A **Trace Turn** may contain Voice Runtime, Agent Control, Robot Control, and MCP **Trace Spans** under one correlated tree.
- **Voice Metrics** are summary timing records; **Process Trace** is the detailed span/event record for debugging, bottleneck analysis, and future visualization.

## Example dialogue

> **Dev:** "Should Robot Call Validation decide whether a whole pick-and-place task is safe?"
> **Domain expert:** "No. **Robot Call Validation** rejects malformed or unsupported tool calls. **Task Policy Layer** handles obvious step preconditions. **MoveIt Safety Boundary** owns movement safety."

## Flagged ambiguities

- "Safety Coverage" previously implied local movement-safety enforcement; resolved: movement safety means the **MoveIt Safety Boundary**, while **Robot Call Validation** is ergonomic validation only.
- "Motion Safety Layer" is ambiguous; resolved: use **Robot Call Validation** for local tool-call validation and **MoveIt Safety Boundary** for movement safety.
- Robot-side policy, context, validation, and adapter ownership is resolved to the **Robot Control Module**.
- "Metrics" can mean summary turn timing or detailed process tracing; resolved: use **Voice Metrics** for summary timing and **Process Trace** for correlated spans/events.
- **Shared Geometry Model** detail is partially resolved: abstract primitives, transforms, geometric features, and constraints are primary; exact render/planning geometry should be derived or referenced unless a later design decision changes this.
- "TCP coordinates from the hologram" is ambiguous; resolved: the hologram provides a desired object pose, while the planner derives the TCP/release pose.
- "Pick then place" is ambiguous for hologram-guided relocation; resolved: use one **Geometry-Grounded Pick-Place** compound task rather than a two-part sequence.
- "Pick up" is resolved as **Hold Compound Goal** with a post-grasp lift, not as a separate `pick` compound goal.
- Near-term manipulation backend is resolved: the **Staged MoveIt Manipulation Backend** is a first-class explicit backend for the **Task-Level Manipulation Planner**, not an MTC implementation and not a hidden fallback.
- Model-visible manipulation tool name is resolved: use `moveit_plan_manipulation_task`; keep `moveit_plan_compound_task` only as a migration/internal compatibility name until removed.
- Robustness goal is resolved: staged manipulation should be **Recovery-Oriented Manipulation**, meaning the planner tries multiple valid candidates before returning failure and reports the exact failed stage and tried candidates.
- Recovery boundary is resolved: broad recovery happens during planning; execution may retry the same approved stage shape, but semantic replanning after gripper close, attach, or object motion requires a new approved **Task Solution**.
- Attempt semantics are resolved: "more attempts" means **Structured Candidate Search** with a bounded candidate budget, not blind repetition of the same failed grasp or motion request.
- Motion planner choice is resolved: use **Hybrid Manipulation Stage Planning** by default; v1 uses `free_motion` for far approach/travel and reserves Cartesian planning for contact-sensitive local motion.
- Sampled motion scope is resolved: `sampled_motion` is not part of the first hybrid manipulation optimization because it is not yet a complete task-stage planner.
- Hybrid candidate failure is resolved: failure in either the `free_motion` far approach or a following Cartesian local stage counts as a normal candidate failure before the backend asks a **Manipulation Recovery Question**.
- Hybrid preview evidence is resolved: both free-motion and Cartesian motion stages must publish or verify `AgentPath` preview evidence for motion-bearing manipulation success.
- Hybrid failure diagnostics are resolved: failed-candidate summaries include the planner used for each stage so the agent can distinguish far-approach failures from contact-sensitive Cartesian failures.
- Hybrid task-solution strictness is resolved: a hybrid manipulation plan returns a `task_solution_id` only after all required free-motion and Cartesian stages plan successfully.
- Hybrid geometry scope is resolved: the first optimization keeps existing grasp and place candidate geometry and changes only planner choice per stage.
- Hybrid goal scope is resolved: the stage policy applies to `hold`, `place`, `move_and_release`, and `pick_place` wherever those goals include far approach or held-object travel stages.
- Hybrid timeout scope is resolved: v1 keeps the same per-stage timeout while changing planner choice; timeout tuning waits for measurement after the planner split.
- Candidate budget is resolved: v1 staged manipulation may try up to 8 grasp candidates and up to 4 motion attempts per candidate.
- Manipulation goals are resolved: `moveit_plan_manipulation_task` supports `hold`, `place`, `release`, `move_and_release`, and `pick_place`; natural "pick up" maps to `hold`.
- Beam grasp strategy is resolved: horizontal beams use top grasp only unless the user explicitly asks for or approves side recovery; vertical beams use side grasps and may try all four side faces within budget.
- Manipulation sync is resolved: execution must verify relevant held/attached state against current MCP/MoveIt evidence, not trust stale **Robot Context** alone.
- Manipulation failures are resolved: failures return **Manipulation Failure Feedback** with exact stage, code, tried candidates, proven facts, uncertainty, suggested next action, and a human-facing message.
- Recovery questions are resolved: after all clearly allowed automatic candidates are exhausted, Agent Orchestration asks a **Manipulation Recovery Question** instead of silently trying a riskier or semantically different option.
- Manipulation plan success is resolved: required execution evidence is strict, but advisory integration facts and optional pose-update proof must not make an otherwise executable plan fail.
- Manipulation planning observation is resolved: `moveit_plan_manipulation_task` fetches fresh object context and robot-state evidence itself; earlier agent observations are advisory only.
- AR preview naming is resolved: the primary planned manipulation path is named `AgentPath`, with ordered stage names available for debugging.
- AR execution naming is resolved: AR execute publishes payload `AgentPath` to `/UR10/command/execute`; for manipulation this means execute the whole approved goal, not a single staged segment.
- AR stop is resolved: `/UR10/command/stop` with payload `AgentPath` cancels the active AgentPath task, invalidates cached execution, and requires fresh observation plus a new plan before continuation.
- AR gripper control is resolved: `/Robot/gripper` is a debug-only operator surface for this workflow; it is not normal task execution, HITL recovery, attachment proof, or release proof.
- Preview/approval boundary is resolved: visible preview is required before `ok=true`, but physical execution still requires explicit spoken approval or an explicit AR execute action bound to the current `AgentPath` and cached **Task Solution**.
- Partial planning is resolved: if any required task stage cannot be planned or previewed, planning returns `ok=false` with no `task_solution_id`; partial stages are diagnostics only.
- Hold lift ownership is resolved: Agent Orchestration supplies `lift_distance_m`; prompt default is `0.10` m, v1 bounds are `0.03`-`0.20` m, and Robot Control rejects out-of-bounds values.
- AR preview shape is resolved: composed **AR Planned Trajectory Preview** is preferred, explicit staged preview publishes one ordered `PlannedTrajectory` per motion stage when composition is unavailable.
- AR preview publication failure is resolved: publication failure for a motion-bearing task fails planning, while zero AR subscribers remain advisory.
- Grasp control is resolved: Agent Orchestration may pass grasp-face preferences when fresh object context makes the choice clear; the staged backend still filters impossible beam faces, ranks candidates from scene evidence, and reports selected-candidate evidence.
- Task solution cache ownership is resolved: MCP owns immutable cached **Task Solution** payloads keyed by `task_solution_id`.
- Task solution freshness is resolved: executing a **Task Solution** requires matching normalized relevant-scene **Scene Snapshot Evidence**, Robot Control/MCP execution-time hash recomputation, and approval within 60 seconds or the current spoken approval turn.
- Failure taxonomy is resolved: compound-task failures use stable **MTC Failure Code** values with structured correction fields.
- AR preview publisher ownership is resolved: `/vizor_robot_control` remains the **AR Planned Trajectory Preview** publisher authority.
- Plain release preview is resolved: **Release Compound Goal** reports no-motion preview evidence instead of publishing `/UR10/request/planned_path`.
- Release-to-physical-model update is resolved: release updates the **Physical Geometry Model** only when release proof includes **Full Object Pose Evidence**.
- Execution contract completeness is resolved: no typed handler or proof requirement means no executable **Task Solution**.
- Hologram target lookup is resolved as **Geometry World Context** injection, not an extra target-pose tool call or freeform prompt-file reading.
- Missing hologram target data is resolved as a hard blocker, not an opportunity to infer a fallback target.
- "Use TCP pose to update the physical model" is ambiguous; resolved: raw TCP pose is not an element pose and must not directly mutate the **Physical Geometry Model**.
- "Frame conversion between Grasshopper and RViz" is resolved as **Calibrated Workspace Coordinates**; no transform is applied while the Grasshopper-to-RViz component keeps coordinates aligned.
- `element_01` versus `dynamic_01` is resolved: use unpadded **Canonical Dynamic Name** such as `dynamic_1`; accept padded `dynamic_01` only at input boundaries when normalized.
- "Drop it" is resolved as **Release Intent** for the currently held or attached object, not an uncontrolled physical drop.

## Current limitation

Emergency stop is currently a Runtime Profile scaffold and detector configuration holder. It does not implement a runtime audio bypass or preemptive stop path.
