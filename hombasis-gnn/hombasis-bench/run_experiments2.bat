@echo off
REM This script runs your commands for each seed in (41, 95, 12, 35)

for %%S in (41 95 12 35) do (
    echo ------------------------------------
    echo Running experiments with seed %%S
    echo ------------------------------------

    echo Running GAT_100k_rd_bc base
    python run-biconn.py -c .\config\biconn\gat-100-base.yaml -group GAT_100k_rd_bc -project GRL -seed %%S

    echo Running GCN_100k_rd_bc base
    python run-biconn.py -c .\config\biconn\gcn-100-base.yaml -group GCN_100k_rd_bc -project GRL -seed %%S

    echo Running GIN_100k_rd_bc base
    python run-biconn.py -c .\config\biconn\gin-100-base.yaml -group GIN_100k_rd_bc -project GRL -seed %%S
)
