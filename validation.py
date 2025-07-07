import matplotlib.pyplot as plt
import pandas as pd

# Literature reflectance values
literature_reflectance = [
    {"method": "Particle Swarm Optimization (2018)", "reflectance": 4.5, "ref": "Khezripour et al. 2018"},
    {"method": "RCWA (2008)", "reflectance": 2.5, "ref": "Sun et al. 2008"},
    {"method": "RCWA Simulation (2015)", "reflectance": 3.0, "ref": "Dong et al. 2015"},
    {"method": "Hybrid Coating (2014)", "reflectance": 10.0, "ref": "Kubota et al. 2014"},
    {"method": "Lithography (2014)", "reflectance": 12.0, "ref": "Xu et al. 2014"},
    {"method": "Electromagnetic Simulation (2014)", "reflectance": 3.0, "ref": "Yuan et al. 2014"},
    {"method": "Numerical Modeling (2024)", "reflectance": 6.0, "ref": "Papatzacos et al. 2024"},
    {"method": "Nanoimprint Lithography (2012)", "reflectance": 5.0, "ref": "Tommila et al. 2012"},
    {"method": "Advanced Meshing (2017)", "reflectance": 4.0, "ref": "Tan et al. 2017"},
    {"method": "Parameter Optimization (2011)", "reflectance": 1.5, "ref": "Yamada et al. 2011"}
]

def plot_literature_comparison(methods: list, reflectances: list) -> None:
    """
    Plot a comparison of reflectance values from literature and this work.
    Args:
        methods (list): List of method names.
        reflectances (list): List of reflectance values.
    Returns:
        None
    """
    methods = [d['method'] for d in literature_reflectance] + methods
    reflectance = [d['reflectance'] for d in literature_reflectance] + reflectances
    colors = ['b','g','purple','r','orange','c','m','brown','gray','k'] + ['#1f77b4']*len(methods)
    plt.figure(figsize=(12,6))
    plt.bar(methods, reflectance, color=colors[:len(methods)])
    plt.ylabel('Reflectance (%)')
    plt.title('Comparison of Reflectance Reduction Methods from Literature and Simulation')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('results/literature_comparison.png')
    plt.close()

def export_literature_comparison(methods: list, reflectances: list) -> None:
    """
    Export literature comparison data to a CSV or text file.
    Args:
        methods (list): List of method names.
        reflectances (list): List of reflectance values.
    Returns:
        None
    """
    rows = [
        {'Method': d['method'], 'Reflectance (%)': d['reflectance'], 'Reference': d['ref']}
        for d in literature_reflectance
    ]
    for m, r in zip(methods, reflectances):
        rows.append({'Method': m, 'Reflectance (%)': r, 'Reference': 'This work'})
    df = pd.DataFrame(rows)
    df.to_csv('results/literature_comparison.csv', index=False)
    with open('results/literature_comparison.tex', 'w') as f:
        f.write(df.to_latex(index=False, float_format='%.2f')) 