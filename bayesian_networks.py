import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
import itertools

np.random.seed(7)

def build_model():
    model = DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])

    cpd_cloudy = TabularCPD("Cloudy", 2, values=[[0.5], [0.5]])
    cpd_rain = TabularCPD("Rain", 2, values=[[0.8, 0.2], [0.2, 0.8]], 
                          evidence=["Cloudy"], evidence_card=[2])
    cpd_sprinkler = TabularCPD("Sprinkler", 2, values=[[0.5, 0.9], [0.5, 0.1]], 
                               evidence=["Cloudy"], evidence_card=[2])
    cpd_wetgrass = TabularCPD("WetGrass", 2, 
                              values=[[0.99, 0.10, 0.10, 0.01], [0.01, 0.90, 0.90, 0.99]],
                              evidence=["Rain", "Sprinkler"], evidence_card=[2, 2])

    model.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler, cpd_wetgrass)
    model.check_model()
    return model

model = build_model()
print("✓ Model built")

# Query 1: P(S=1 | W=1)
inference = VariableElimination(model)

query1 = inference.query(variables=["Sprinkler"], evidence={"WetGrass": 1}, show_progress=False)
print(f"\nP(Sprinkler=1 | WetGrass=1) = {query1.values[1]:.4f}")

# Query 2: P(C=1 | W=1)
query2 = inference.query(variables=["Cloudy"], evidence={"WetGrass": 1}, show_progress=False)
print(f"P(Cloudy=1 | WetGrass=1) = {query2.values[1]:.4f}")

# Query 3: P(R=1 | W=1, S=0)
query3 = inference.query(variables=["Rain"], evidence={"WetGrass": 1, "Sprinkler": 0}, show_progress=False)
print(f"P(Rain=1 | WetGrass=1, Sprinkler=0) = {query3.values[1]:.4f}")

# CPT structure
print("\n" + "="*50)
print("WetGrass CPD:")
print(model.get_cpds("WetGrass"))
print("\nColumn order: [R=0,S=0], [R=0,S=1], [R=1,S=0], [R=1,S=1]")

# Sampling variability demo
print("\n" + "="*50)
print("Sampling Variability (N=100):")

def make_model():
    return DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])

results = []
for seed in [1, 2, 3, 4, 5, 42]:
    data = model.simulate(n_samples=100, seed=seed, show_progress=False)
    m = make_model()
    
    cpd_cloudy = TabularCPD("Cloudy", 2, values=[[0.5], [0.5]])
    cpd_rain = TabularCPD("Rain", 2, values=[[0.8, 0.2], [0.2, 0.8]], 
                          evidence=["Cloudy"], evidence_card=[2])
    cpd_sprinkler = TabularCPD("Sprinkler", 2, values=[[0.5, 0.9], [0.5, 0.1]], 
                               evidence=["Cloudy"], evidence_card=[2])
    cpd_wetgrass = TabularCPD("WetGrass", 2, 
                              values=[[0.99, 0.10, 0.10, 0.01], [0.01, 0.90, 0.90, 0.99]],
                              evidence=["Rain", "Sprinkler"], evidence_card=[2, 2])
    m.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler, cpd_wetgrass)
    m.fit(data, estimator=MaximumLikelihoodEstimator)
    
    rain_cpd = m.get_cpds("Rain")
    estimate = float(rain_cpd.get_value(Rain=1, Cloudy=1))
    results.append({'seed': seed, 'estimate': estimate, 'error': abs(estimate - 0.8)})

df = pd.DataFrame(results)
print(df.to_string(index=False))
print(f"\nTrue P(R=1|C=1) = 0.8, different seeds give different estimates")

# Convergence
print("\n" + "="*50)
print("Convergence with Sample Size:")

results_by_n = []
for n in [20, 50, 100, 200, 500, 1000]:
    data = model.simulate(n_samples=n, seed=99, show_progress=False)
    m = make_model()
    
    cpd_cloudy = TabularCPD("Cloudy", 2, values=[[0.5], [0.5]])
    cpd_rain = TabularCPD("Rain", 2, values=[[0.8, 0.2], [0.2, 0.8]], 
                          evidence=["Cloudy"], evidence_card=[2])
    cpd_sprinkler = TabularCPD("Sprinkler", 2, values=[[0.5, 0.9], [0.5, 0.1]], 
                               evidence=["Cloudy"], evidence_card=[2])
    cpd_wetgrass = TabularCPD("WetGrass", 2, 
                              values=[[0.99, 0.10, 0.10, 0.01], [0.01, 0.90, 0.90, 0.99]],
                              evidence=["Rain", "Sprinkler"], evidence_card=[2, 2])
    m.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler, cpd_wetgrass)
    m.fit(data, estimator=MaximumLikelihoodEstimator)
    
    rain_cpd = m.get_cpds("Rain")
    estimate = float(rain_cpd.get_value(Rain=1, Cloudy=1))
    results_by_n.append({'N': n, 'estimate': estimate, 'error': abs(estimate - 0.8)})

df_n = pd.DataFrame(results_by_n)
print(df_n.to_string(index=False))

# Validation
print("\n" + "="*50)
print("Independent Verification:")

def joint_prob(c, r, s, w):
    p_c = 0.5
    p_r = 0.8 if r == 1 else 0.2 if c == 1 else 0.2
    if c == 1:
        p_r = 0.8 if r == 1 else 0.2
    else:
        p_r = 0.2 if r == 1 else 0.8
    
    p_s = 0.5 if s == 1 else 0.5 if c == 1 else 0.1 if s == 1 else 0.9
    if c == 1:
        p_s = 0.5 if s == 1 else 0.5
    else:
        p_s = 0.1 if s == 1 else 0.9
    
    p_w_table = {(0, 0): 0.01, (0, 1): 0.90, (1, 0): 0.90, (1, 1): 0.99}
    p_w = p_w_table[(r, s)] if w == 1 else 1 - p_w_table[(r, s)]
    
    return p_c * p_r * p_s * p_w

numerator = sum(joint_prob(c, r, s, w) for c, r, s, w in itertools.product([0, 1], repeat=4) 
                if w == 1 and r == 1)
denominator = sum(joint_prob(c, r, s, w) for c, r, s, w in itertools.product([0, 1], repeat=4) 
                  if w == 1)

manual = numerator / denominator
pgmpy_val = float(inference.query(["Rain"], evidence={"WetGrass": 1}, show_progress=False).values[1])

print(f"P(Rain=1|WetGrass=1):")
print(f"  Manual calculation: {manual:.6f}")
print(f"  pgmpy result: {pgmpy_val:.6f}")
print(f"  Match: {'✓' if abs(manual - pgmpy_val) < 1e-10 else '✗'}")
