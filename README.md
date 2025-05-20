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
                  --min-payload 1000 --max-payload 1500
```

El script genera:

* `out/tsn_schedule_UNIDIR_30.html` → visualización Plotly  
* `out/UNIDIR/UNIDIR.{ned,ini}`    → simulación OMNeT++  

## Entrenamiento (opcional)

```bash
python ui/train.py --time_steps 50000 --topo UNIDIR --num_flows 49 --link_rate 1000 --min-payload 800 --max-payload 1500
```




