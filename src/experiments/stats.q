tbl:("SSIIF";enlist ",") 0: `:result_table.csv;
tbl:select from tbl where seed<10;
tbl:update scaled_reward:reward%nsteps from tbl;
tbl:select avg scaled_reward by agent, seed, nsteps from tbl;
tbl:select reward:avg scaled_reward, stderr:(sdev scaled_reward)%count[i] by agent, nsteps from tbl;
// Below, switch nsteps to any other desired number
select from tbl where nsteps=500