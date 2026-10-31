# fall-2026-nuclear-power-for-data-centers
Team project: fall-2026-nuclear-power-for-data-centers

Data centers need large amounts of power around the clock, and nuclear reactors are one of the few sources that can supply it steadily. Reactors still go offline, though, for scheduled refueling, maintenance and unexpected faults. When they do, the missing output has to come from somewhere else.

This project uses 25+ years of US Nuclear Regulatory Commission (NRC) data to measure how reliably each US reactor delivers power and to forecast its output.

## The fleet

We study the **94 currently operating US commercial reactors**, grouped into the NRC's four regions.

![Operating US nuclear reactors by NRC region](reports/reactors_by_region.png)

## Data

All data comes from the NRC:

| Source | What it gives us |
|---|---|
| [NRC reactor facility finder](https://www.nrc.gov/facilities-safety/facility-finder/reactors) | Each operating reactor's name, NRC region and state |
| [NRC power reactor status reports](https://www.nrc.gov/documents-reports/document-collections/events-reports-associated-with/power-reactor-status-reports/) | Each reactor's daily power as a percentage of full power from 2000 to 2025, and the reason whenever it is below 100% |
