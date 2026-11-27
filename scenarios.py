"""Vistas pública/privada; no devolver un caso mixto al agente."""
from data_generation.build import make_dataset

def build_cases(seed=42):
    return make_dataset(seed=seed)
