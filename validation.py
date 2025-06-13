import matplotlib.pyplot as plt
import pandas as pd

# Literature reflectance values (from slide image)
literature_reflectance = [
    {"method": "Particle Swarm (2018)", "reflectance": 4.5, "ref": "Khezripour et al. 2018"},
    {"method": "RCWA (2008)", "reflectance": 2.5, "ref": "Sun et al. 2008"},
    {"method": "FDTD (2005)", "reflectance": 2.8, "ref": "Dong et al. 2015"},
    {"method": "Hybrid Coating (2014)", "reflectance": 10.0, "ref": "Yuan et al. 2014"},
    {"method": "Lithography (2014)", "reflectance": 12.0, "ref": "Yuan et al. 2014"},
    {"method": "Electromagnetic Sim. (2014)", "reflectance": 3.0, "ref": "Yuan et al. 2014"},
    {"method": "Numerical Modeling (2024)", "reflectance": 6.0, "ref": "Recent work"},
    {"method": "Nanoimprint Litho. (2021)", "reflectance": 5.0, "ref": "Recent work"},
    {"method": "Advanced Meshing (2017)", "reflectance": 4.0, "ref": "Recent work"},
    {"method": "Parameter Optimization (2011)", "reflectance": 1.5, "ref": "Recent work"},
]

def plot_literature_comparison(simulated_methods, simulated_reflectance, fname='results/literature_comparison.png'):
    methods = [d['method'] for d in literature_reflectance] + simulated_methods
    reflectance = [d['reflectance'] for d in literature_reflectance] + simulated_reflectance
    colors = ['b','g','purple','r','orange','c','m','brown','gray','k'] + ['#1f77b4']*len(simulated_methods)
    plt.figure(figsize=(12,6))
    plt.bar(methods, reflectance, color=colors[:len(methods)])
    plt.ylabel('Reflectance (%)')
    plt.title('Comparison of Reflectance Reduction Methods from Literature and Simulation')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def export_literature_comparison(simulated_methods, simulated_reflectance, fname_csv='results/literature_comparison.csv', fname_tex='results/literature_comparison.tex'):
    rows = [
        {'Method': d['method'], 'Reflectance (%)': d['reflectance'], 'Reference': d['ref']}
        for d in literature_reflectance
    ]
    for m, r in zip(simulated_methods, simulated_reflectance):
        rows.append({'Method': m, 'Reflectance (%)': r, 'Reference': 'This work'})
    df = pd.DataFrame(rows)
    df.to_csv(fname_csv, index=False)
    with open(fname_tex, 'w') as f:
        f.write(df.to_latex(index=False, float_format='%.2f')) 