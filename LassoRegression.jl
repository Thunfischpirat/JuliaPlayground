# Load admission data from Data/Admission_data/adm_data.csv

using CSV
using DataFrames
using LinearAlgebra
using Plots

# Load data
df = CSV.read("Data/Admission_data/adm_data.csv", DataFrame)