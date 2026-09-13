# Repository guidance

- Use generic, formal repository/module/class names. Model sizes, context lengths and format revisions belong in configuration or metadata.
- Keep the policy, optimization, environment adapter, timing model and artifact handling independent. The project has no RSL-RL dependency.
- Treat `previous_issued_action`, transport arrival, controller application and physical motor response as distinct events.
- Hardware timing, PID parameters and robot assets require their own evidence. The existing `isaac_wheeled_rl_deploy` workspace is a demo, not a verified deployment implementation.
- This implementation phase permits unit/interface checks with synthetic tensors. Robot training and simulator runs require a new user instruction.
- Code, identifiers and comments are English; explanatory Markdown is Chinese.
- Automated commits in this workspace must use author and committer `yukikaze2233 <yingziyuw@gmail.com>`. Set the four `GIT_AUTHOR_*` / `GIT_COMMITTER_*` environment variables for the commit rather than modifying global Git configuration.
- Run relevant tests, and preserve other contributors' work. Do not commit generated models, run data or local environments.
