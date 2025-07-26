# 2025-07-23 23:57:26
using meta-llama_llama-3_2-3b-instruct

aggregated_score_raw
    Final Judgemark (raw)   = 0.088
    Final Judgemark (cal)  = 0.089
    

aggregated_score_weighted
        Final Judgemark (raw)   = 0.089
    Final Judgemark (cal)  = 0.102
    
aggregated_score_ranked
        Final Judgemark (raw)   = 0.034
    Final Judgemark (cal)  = 0.030
# 

Hm maybe I should try without p value, maybe with restricted scale
got to load the saved logp from the runfile

# 2025-07-25 19:30:21

Withdeepseek

Normed logp
Final Judgemark (raw)   = 0.673
Final Judgemark (cal)  = 0.736

Weighted
Final Judgemark (raw)   = 0.635
Final Judgemark (cal)  = 0.660

argmax
Final Judgemark (raw)   = 0.635
Final Judgemark (cal)  = 0.659

normal ranked (without stretching)

normed and weighted
Final Judgemark (raw)   = 0.624
Final Judgemark (cal)  = 0.645



| name          | score    | score_norm |
|---------------|----------|------------|
| ranked_scaled | 0.67     |     0.79   |
| ranked_norm   | 0.67     |     0.73   |
| weighted      | 0.63     |     0.65   |
| raw           | 0.63     |     0.65   |
| weighted_norm | 0.62     |     0.64   |
| ranked        | 0.33     |     0.28   |
