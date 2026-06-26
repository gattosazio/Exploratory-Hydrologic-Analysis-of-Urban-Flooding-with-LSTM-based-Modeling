# Hydrologic Findings Summary

The selected-event descriptive analysis should focus on hydrologic behavior rather than on graph volume. The selected rainfall events show recurring delayed canal response after rainfall-intensity peaks. Across the selected events, Node 1 has a median peak-to-peak lag of 3.00 h, while Node 2 has a median peak-to-peak lag of 2.50 h. The first-response lag remains close to 2 h for both nodes, supporting the interpretation that the initial canal response commonly occurs within the 2-3 h window.

The selected event set is dominated by moderate-to-heavy events: Light Rain = 0, Moderate Rain = 4, and Heavy Rain = 6. This means the existing event dataset is strongest for explaining moderate-to-heavy rainfall response, while redeployed dry-period and light-rain datasets are needed to strengthen low-intensity baseline interpretation.

The merged selected-event figures show that Node 2 often has stronger rain-intensity peaks, while both nodes can display delayed canal response. Long peak lags occur mainly when rainfall arrives in repeated pulses. These long-lag events should be treated as hydrologic exceptions or compound-pulse responses, not as the typical response pattern.

For LSTM modeling, the key implication is that same-hour rainfall variables are insufficient. The model should include antecedent rainfall intensity, accumulated rainfall, prior canal water level, and lagged variables. A 2-3 h horizon is defendable for first response, while longer horizons should be evaluated for peak water-level prediction.
