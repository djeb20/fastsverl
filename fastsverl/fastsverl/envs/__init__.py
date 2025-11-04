from gymnasium.envs.registration import register

register(
    id='Mastermind-v0',
    entry_point='fastsverl.envs.mastermind:Mastermind',
)

register(
    id='FactoredTaxi-v3',
    entry_point='fastsverl.envs.factored_taxi:FactoredTaxi',
)

register(
    id='GWB-v0',
    entry_point='fastsverl.envs.gwb:GWB',
)

register(
    id='ShipNav-v0',
    entry_point='fastsverl.envs.shipnav:ShipNav',
)

register(
    id='Hypercube-v0',
    entry_point='fastsverl.envs.hypercube:Hypercube',
)

register(
    id='FlatMinigrid-v0',
    entry_point='fastsverl.envs.flat_minigrid:FlatMinigrid',
)