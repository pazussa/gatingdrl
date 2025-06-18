# gatingdrl – Scheduling TSN con Deep RL

Proyecto para generar Gate Control Lists en redes
Time-Sensitive Networking (TSN) mediante **Maskable PPO**.  

Compatible Ubuntu 20.04/22.04 – Python 3.10+.  

## Instalación rápida

```bash
# 1) clona y entra en el repo
git clone https://github.com/tu-usuario/gatingdrl.git
cd gatingdrl

# 2) entorno virtual
python3.10 -m venv venv
source venv/bin/activate
export PYTHONPATH=.

# 3) deps CPU (estándar) o GPU (CUDA 11.8)
pip install --upgrade pip
pip install -e .

```

## Prueba 

```bash
python ui/test.py --topo UNIDIR --num_flows 30 \
                  --alg MaskablePPO --link_rate 1000 \
                  --min-payload 1000 --max-payload 1500 --seed 42
```

El script genera:

* `out/tsn_schedule_UNIDIR_30.html` → visualización Plotly  
* `out/UNIDIR/UNIDIR.{ned,ini}`    → simulación OMNeT++  

## Entrenamiento (opcional)

```bash
python ui/train.py --time_steps 50000 --topo UNIDIR --num_flows 49 --link_rate 1000 --min-payload 800 --max-payload 1500 --seed 42
```

# GatingDRL: TSN Scheduling with Deep Reinforcement Learning

## Overview

This project implements Time Sensitive Networks (TSN) scheduling using Deep Reinforcement Learning, specifically MaskablePPO from Stable Baselines3. The system optimizes gate control schedules for time-aware traffic shaping in industrial networks.

## Features

- **Multiple Network Topologies**: Support for various network configurations (SIMPLE, UNIDIR, etc.)
- **Deep Reinforcement Learning**: Uses MaskablePPO for intelligent scheduling decisions
- **Interactive Visualization**: Plotly-based interactive scheduling visualizations
- **OMNeT++ Export**: Generates .ned and .ini files for network simulation
- **Comprehensive Metrics**: Latency analysis, link utilization, and convergence tracking
- **Reproducible Results**: Seed support for deterministic experiments

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```bash
python ui/train.py --time_steps 100000 --num_flows 50 --topo UNIDIR8 --seed 42
```

### Testing a Model

```bash
python ui/test.py --topo UNIDIR8 --num_flows 100 --seed 42
```

## Reproducibility

Use the `--seed` parameter to ensure reproducible results:

```bash
# Training with reproducible results
python ui/train.py --time_steps 50000 --topo UNIDIR --num_flows 49 \
                   --link_rate 1000 --min-payload 800 --max-payload 1500 \
                   --seed 42
```

Curriculum learning is disabled by default for more stable training. Use `--curriculum` to gradually increase difficulty.

```bash
# Testing with same seed
python ui/test.py --topo UNIDIR --num_flows 30 \
                  --alg MaskablePPO --link_rate 1000 \
                  --min-payload 1000 --max-payload 1500 \
                  --seed 42
```

## Project Structure




