# project-scalion

## Setup

```bash
pip install browsergym  # (recommended) everything below
pip install browsergym-experiments  # experiment utilities (agent, loop, benchmarks) + everything below
pip install browsergym-core  # core functionalities only (no benchmark, just the openended task)
pip install browsergym-miniwob  # core + miniwob
pip install browsergym-webarena  # core + webarena
pip install browsergym-visualwebarena  # core + visualwebarena
pip install browsergym-workarena  # core + workarena
pip install browsergym-assistantbench  # core + assistantbench
pip install weblinx-browsergym  # core + weblinx
```

```bash
playwright install chromium
```

## Run Web Agent

```bash
cd demo_agent
export MAP="https://www.openstreetmap.org/#map=5/38.01/-95.84"
python run_demo.py --start_url $MAP
# instruction: "Tell me the coordinates of Apple Store near Pitt in DD format"
```

## Summarize Search Strategies

```bash
pip install litellm
```
