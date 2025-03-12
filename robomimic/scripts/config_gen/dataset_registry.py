from collections import OrderedDict
from copy import deepcopy
import os

########################################

robocasa_hitl_tasks = [
    "CloseDoorSingleHinge",
    "OpenDoorDoubleHinge",
    "OpenDoorSingleHinge",
    "TurnOnSinkFaucet",
    "TurnOnMicrowave",
    "TurnOffSinkFaucet",
    "TurnOffMicrowave",
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "CoffeeSetupMug",
    "CoffeeServeMug"
]

mutex_hitl_tasks = [
    "rw2_put_the_blue_cup_in_the_basket",                                  
    "rw5_put_the_blue_mug_on_the_oven_tray",
    "rw7_put_the_book_in_the_back_compartment_of_the_caddy",
    "rw2_put_the_red_bowl_in_the_basket",
    "rw5_put_the_bread_on_the_white_plate",
    "rw7_put_the_red_cup_in_the_front_compartment_of_the_caddy",
    "rw4_pull_out_the_tray_of_the_oven_and_put_the_bowl_with_hot_dogs_on_the_tray",
    "rw6_put_the_bowl_with_hot_dogs_on_the_white_plate",
    "rw4_put_the_book_in_the_back_compartment_of_the_caddy",
    "rw6_put_the_pink_mug_on_the_white_plate",
]

robocasa_demo_path = "~/robocasa"
robocasa_round01_path = "~/robocasa-hitl/round01"
robocasa_round012_path = "~/robocasa-hitl/round012"
mutex_demo_path = "~/mutex"
mutex_round01_path = "~/mutex-hitl/round01"
mutex_round012_path = "~/mutex-hitl/round01"

########################################

ROBOCASA_ORIGINAL = OrderedDict(
    PnPCounterToCab=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["condiment_bottle", "baguette", "kettle_electric", "avocado", "can"], exclude_obj_groups=None)),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/2024-01-21",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-01-22-07-53-41",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-01-28-07-25-32",
        mg_5scenes_filter_key="train_2500_demos",
        mg_16tasks_path=[
            "~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-01-29-10-17-48/demo_gentex_im128_subset1.hdf5",
            "~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-01-29-10-17-48/demo_gentex_im128_subset2.hdf5",
        ],
        mg_16tasks_filter_key="train_20000_demos",
        mg_5scenes_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-01-31-06-03-46",
        mg_5scenes_16tasks_filter_key="train_20000_demos",
        mg_6objcats_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-02-06-07-50-17",
        mg_6objcats_filter_key="train_2500_demos",
        horizon=500,
    ),
    PnPCabToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["beer", "orange", "jam", "canned_food", "coffee_cup"], exclude_obj_groups=None)),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCabToCounter/2024-01-21",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-01-22-08-04-03",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-01-28-07-26-19",
        mg_5scenes_filter_key="train_2500_demos",
        mg_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-01-29-10-17-52",
        mg_16tasks_filter_key="train_6000_demos",
        mg_5scenes_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-01-31-06-05-09",
        mg_5scenes_16tasks_filter_key="train_6000_demos",
        mg_6objcats_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-02-06-07-50-45",
        mg_6objcats_filter_key="train_2500_demos",
        horizon=500,
    ),
    PnPCounterToSink=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["apple", "banana", "bar_soap", "cup", "cucumber"], exclude_obj_groups=None)),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToSink/2024-01-21",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-01-22-02-46-24",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-01-28-07-26-40",
        mg_5scenes_filter_key="train_2500_demos",
        mg_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-01-29-10-18-04",
        mg_16tasks_filter_key="train_9000_demos",
        mg_5scenes_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-01-31-06-07-09",
        mg_5scenes_16tasks_filter_key="train_9000_demos",
        mg_6objcats_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-02-06-07-51-21",
        mg_6objcats_filter_key="train_2500_demos",
        horizon=500,
    ),
    PnPSinkToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["peach", "lime", "yogurt", "fish", "kiwi"], exclude_obj_groups=None)),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPSinkToCounter/2024-01-21",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-01-22-02-46-07",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-01-28-07-27-15",
        mg_5scenes_filter_key="train_2500_demos",
        mg_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-01-29-10-18-07",
        mg_16tasks_filter_key="train_6000_demos",
        mg_5scenes_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-01-31-06-11-24",
        mg_5scenes_16tasks_filter_key="train_6000_demos",
        mg_6objcats_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-02-06-07-51-39",
        mg_6objcats_filter_key="train_2500_demos",
        horizon=500,
    ),
   PnPCounterToStove=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["potato", "garlic", "steak", "eggplant", "mango"], exclude_obj_groups=None)),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToStove/2024-01-20",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-01-21-06-50-56",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-01-28-07-29-10",
        mg_5scenes_filter_key="train_2500_demos",
        mg_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-01-29-10-18-13",
        mg_16tasks_filter_key="train_9000_demos",
        mg_5scenes_16tasks_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-01-31-06-07-33",
        mg_5scenes_16tasks_filter_key="train_9000_demos",
        mg_6objcats_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-02-06-07-52-49",
        mg_6objcats_filter_key="train_2500_demos",
        horizon=500,
    ),
    PnPCounterToMicrowave=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["broccoli", "cheese", "bell_pepper", "squash", "sweet_potato"], exclude_obj_groups=None)),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-01-20",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-01-21-06-50-13",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-01-28-07-28-07",
        mg_5scenes_filter_key="train_2500_demos",
        mg_6objcats_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-02-06-07-52-14",
        mg_6objcats_filter_key="train_2500_demos",
        horizon=500,
    ),
    PnPMicrowaveToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["corn", "tomato", "hot_dog", "egg", "carrot"], exclude_obj_groups=None)),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-01-21",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-01-22-02-45-18",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-01-28-07-28-58",
        mg_5scenes_filter_key="train_2500_demos",
        mg_6objcats_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-02-06-07-52-39",
        mg_6objcats_filter_key="train_2500_demos",
        horizon=500,
    ),
    OpenDoorDoubleHinge=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenDoorDoubleHinge/2024-01-23",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenDoorDoubleHinge/mg/2024-01-23-20-42-16",
        mg_filter_key="train_1000_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenDoorDoubleHinge/mg/2024-01-28-07-30-24",
        mg_5scenes_filter_key="train_1000_demos",
        horizon=800,
    ),
    CloseDoorDoubleHinge=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseDoorDoubleHinge/2024-01-22",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseDoorDoubleHinge/mg/2024-01-23-06-33-05",
        mg_filter_key="train_1000_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseDoorDoubleHinge/mg/2024-01-28-07-30-44",
        mg_5scenes_filter_key="train_1000_demos",
        horizon=800,
    ),
    OpenDoorSingleHinge=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenDoorSingleHinge/2024-01-23",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenDoorSingleHinge/mg/2024-01-23-20-42-26",
        mg_filter_key="train_1500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenDoorSingleHinge/mg/2024-01-28-07-31-11",
        mg_5scenes_filter_key="train_1500_demos",
        horizon=500,
    ),
    CloseDoorSingleHinge=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseDoorSingleHinge/2024-01-22",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseDoorSingleHinge/mg/2024-01-22-23-55-41",
        mg_filter_key="train_1500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseDoorSingleHinge/mg/2024-01-28-07-31-39",
        mg_5scenes_filter_key="train_1500_demos",
        horizon=500,
    ),
    TurnOnSinkFaucet=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-01-23",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-01-25-09-20-09",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-01-28-07-32-09",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=500,
    ),
    TurnOffSinkFaucet=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-01-23",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-01-25-09-20-24",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-01-28-07-33-13",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=500,
    ),
    TurnOnStove=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOnStove/2024-01-24",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOnStove/mg/2024-01-25-09-20-45",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOnStove/mg/2024-01-28-07-33-39",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=500,
    ),
    TurnOffStove=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOffStove/2024-01-24",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOffStove/mg/2024-01-25-09-21-03",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOffStove/mg/2024-01-28-07-34-04",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=500,
    ),
    CoffeeSetupMug=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeSetupMug/2024-01-24",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-01-25-09-21-40",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-01-28-07-34-36",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=800,
    ),
    CoffeeServeMug=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeServeMug/2024-01-24",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-01-25-09-21-55",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-01-28-07-35-08",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=800,
    ),
    CoffeePressButton=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeePressButton/2024-01-23",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-01-26-09-04-58",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-01-28-07-35-31",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=300,
    ),
    TurnOnMicrowave=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOnMicrowave/2024-01-23",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-01-26-09-04-47",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-01-28-07-36-01",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=300,
    ),
    TurnOffMicrowave=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOffMicrowave/2024-01-23",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-01-26-09-04-50",
        mg_filter_key="train_2500_demos",
        mg_5scenes_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-01-28-07-36-23",
        mg_5scenes_filter_key="train_2500_demos",
        horizon=300,
    ),
)

def create_robocasa_hitl_tasks(robocasa_hitl_path):
    return OrderedDict(
    PnPCounterToCab=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["condiment_bottle", "baguette", "kettle_electric", "avocado", "can"], exclude_obj_groups=None)),
        horizon=500,    
        human_path=f"{robocasa_hitl_path}/PnPCounterToCab",
        human_filter_key=None,
    ),
    
    PnPCabToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["beer", "orange", "jam", "canned_food", "coffee_cup"], exclude_obj_groups=None)),
        horizon=500,
        human_path=f"{robocasa_hitl_path}/PnPCabToCounter",
        human_filter_key=None,
    ),
    PnPCounterToSink=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["apple", "banana", "bar_soap", "cup", "cucumber"], exclude_obj_groups=None)),
        horizon=500,
        human_path=f"{robocasa_hitl_path}/PnPCounterToSink",
        human_filter_key=None,    
    ),
    PnPSinkToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["peach", "lime", "yogurt", "fish", "kiwi"], exclude_obj_groups=None)),
        horizon=500,
        human_path=f"{robocasa_hitl_path}/PnPCounterToSink",
        human_filter_key=None,     
    
    ),
   PnPCounterToStove=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["potato", "garlic", "steak", "eggplant", "mango"], exclude_obj_groups=None)),
        horizon=500,      
        human_path=f"{robocasa_hitl_path}/PnPCounterToStove",
        human_filter_key=None,  
    ),
    PnPCounterToMicrowave=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["broccoli", "cheese", "bell_pepper", "squash", "sweet_potato"], exclude_obj_groups=None)),
        horizon=500,
        human_path=f"{robocasa_hitl_path}/PnPCounterToMicrowave",
        human_filter_key=None,
    ),
    PnPMicrowaveToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict(obj_groups=["corn", "tomato", "hot_dog", "egg", "carrot"], exclude_obj_groups=None)),
        horizon=500,
        human_path=f"{robocasa_hitl_path}/PnPMicrowaveToCounter",
        human_filter_key=None,
    
    ),
    OpenDoorDoubleHinge=dict(
        horizon=800,
        human_path=f"{robocasa_hitl_path}/OpenDoorDoubleHinge",
        human_filter_key=None,
    
    ),
    CloseDoorDoubleHinge=dict(
        horizon=800,
        human_path=f"{robocasa_hitl_path}/CloseDoorDoubleHinge",
        human_filter_key=None,
    ),
    OpenDoorSingleHinge=dict(
        horizon=500,
        human_path=f"{robocasa_hitl_path}/OpenDoorSingleHinge",
        human_filter_key=None,
    ),
    CloseDoorSingleHinge=dict(
        horizon=500,  
        human_path=f"{robocasa_hitl_path}/CloseDoorSingleHinge",
        human_filter_key=None,   
    ),
    TurnOnSinkFaucet=dict(
        horizon=500,
        human_path=f"{robocasa_hitl_path}/TurnOnSinkFaucet",
        human_filter_key=None,  
    ),
    TurnOffSinkFaucet=dict(
        horizon=500,
        human_path=f"{robocasa_hitl_path}/TurnOffSinkFaucet",
        human_filter_key=None,     
    ),
    TurnOnStove=dict(
        horizon=500,
        human_path=f"{robocasa_hitl_path}/TurnOnStove",
        human_filter_key=None,         
    ),
    TurnOffStove=dict(
        horizon=500,
        human_path=f"{robocasa_hitl_path}/TurnOffStove",
        human_filter_key=None,         
    ),
    CoffeeSetupMug=dict(
        horizon=800,
        human_path=f"{robocasa_hitl_path}/CoffeeSetupMug",
        human_filter_key=None,
    ),
    CoffeeServeMug=dict(
        horizon=800,
        human_path=f"{robocasa_hitl_path}/CoffeeServeMug",
        human_filter_key=None,
    ),
    CoffeePressButton=dict(
        horizon=300,
        human_path=f"{robocasa_hitl_path}/CoffeePressButton",
        human_filter_key=None,         
    ),
    TurnOnMicrowave=dict(
        horizon=300,
        human_path=f"{robocasa_hitl_path}/TurnOnMicrowave",
        human_filter_key=None,        
    ),
    TurnOffMicrowave=dict(
        horizon=300,
        human_path=f"{robocasa_hitl_path}/TurnOffMicrowave",
        human_filter_key=None,         
    ),
)

def create_mutex_hitl_tasks(mutex_hitl_path):
    return OrderedDict(
    rw1_open_the_bottom_drawer=dict(
        lang="open the bottom drawer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW1_open_the_bottom_drawer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw1_open_the_top_drawer_and_put_the_bowl_in_the_back_of_the_scene_inside_it=dict(
        lang="open the top drawer and put the bowl in the back of the scene inside it",
        horizon=350,
        mutex_path=f"{mutex_hitl_path}/RW1_open_the_top_drawer_and_put_the_bowl_in_the_back_of_the_scene_inside_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw1_open_the_top_drawer_and_put_the_granola_bar_inside_it=dict(
        lang="open the top drawer and put the granola bar inside it",
        horizon=300,
        mutex_path=f"{mutex_hitl_path}/RW1_open_the_top_drawer_and_put_the_granola_bar_inside_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw1_open_the_top_drawer=dict(
        lang="open the top drawer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW1_open_the_top_drawer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw1_put_the_granola_bar_in_the_back_compartment_of_the_caddy=dict(
        lang="put the granola bar in the back compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW1_put_the_granola_bar_in_the_back_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw1_put_the_granola_bar_in_the_front_compartment_of_the_caddy=dict(
        lang="put the granola bar in the front compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW1_put_the_granola_bar_in_the_front_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw2_pull_out_the_tray_of_the_oven_and_put_the_blue_cup_on_it=dict(
        lang="pull out the tray of the oven and put the blue cup on it",
        horizon=300,
        mutex_path=f"{mutex_hitl_path}/RW2_pull_out_the_tray_of_the_oven_and_put_the_blue_cup_on_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw2_pull_out_the_tray_of_the_oven_and_put_the_red_bowl_on_it=dict(
        lang="pull out the tray of the oven and put the red bowl on it",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW2_pull_out_the_tray_of_the_oven_and_put_the_red_bowl_on_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw2_pull_out_the_tray_of_the_oven=dict(
        lang="pull out the tray of the oven",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW2_pull_out_the_tray_of_the_oven_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw2_put_the_blue_cup_in_the_basket=dict(
        lang="put the blue cup in the basket",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW2_put_the_blue_cup_in_the_basket_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw2_put_the_mac_and_cheese_box_in_the_basket=dict(
        lang="put the mac and cheese box in the basket",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW2_put_the_mac_and_cheese_box_in_the_basket_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw2_put_the_red_bowl_in_the_basket=dict(
        lang="put the red bowl in the basket",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW2_put_the_red_bowl_in_the_basket_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw3_close_the_top_drawer_and_open_the_bottom_drawer=dict(
        lang="close the top drawer and open the bottom drawer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW3_close_the_top_drawer_and_open_the_bottom_drawer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw3_put_the_granola_bar_inside_the_basket=dict(
        lang="put the granola bar inside the basket",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW3_put_the_granola_bar_inside_the_basket_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw3_put_the_granola_bar_inside_the_top_drawer=dict(
        lang="put the granola bar inside the top drawer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW3_put_the_granola_bar_inside_the_top_drawer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw3_put_the_pink_mug_inside_the_basket=dict(
        lang="put the pink mug inside the basket",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW3_put_the_pink_mug_inside_the_basket_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw3_put_the_pink_mug_inside_the_top_drawer_and_close_the_drawer=dict(
        lang="put the pink mug inside the top drawer and close the drawer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW3_put_the_pink_mug_inside_the_top_drawer_and_close_the_drawer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw3_put_the_red_bowl_inside_the_basket=dict(
        lang="put the red bowl inside the basket",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW3_put_the_red_bowl_inside_the_basket_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw3_put_the_red_bowl_inside_the_top_drawer_and_close_the_drawer=dict(
        lang="put the red bowl inside the top drawer and close the drawer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW3_put_the_red_bowl_inside_the_top_drawer_and_close_the_drawer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw4_pull_out_the_tray_of_the_oven_and_put_the_bowl_with_hot_dogs_on_the_tray=dict(
        lang="pull out the tray of the oven and put the bowl with hot dogs on the tray",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW4_pull_out_the_tray_of_the_oven_and_put_the_bowl_with_hot_dogs_on_the_tray_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw4_pull_out_the_tray_of_the_oven_and_put_the_red_cup_on_the_tray=dict(
        lang="pull out the tray of the oven and put the red cup on the tray",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW4_pull_out_the_tray_of_the_oven_and_put_the_red_cup_on_the_tray_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw4_pull_out_the_tray_of_the_oven=dict(
        lang="pull out the tray of the oven",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW4_pull_out_the_tray_of_the_oven_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw4_put_the_book_in_the_back_compartment_of_the_caddy=dict(
        lang="put the book in the back compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW4_put_the_book_in_the_back_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw4_put_the_book_in_the_front_compartment_of_the_caddy=dict(
        lang="put the book in the front compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW4_put_the_book_in_the_front_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw4_put_the_red_cup_in_the_back_compartment_of_the_caddy=dict(
        lang="put the red cup in the back compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW4_put_the_red_cup_in_the_back_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw5_push_the_tray_of_the_oven_in=dict(
        lang="push the tray of the oven in",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW5_push_the_tray_of_the_oven_in_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw5_put_the_blue_mug_on_the_oven_tray=dict(
        lang="put the blue mug on the oven tray",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW5_put_the_blue_mug_on_the_oven_tray_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw5_put_the_blue_mug_on_the_white_plate=dict(
        lang="put the blue mug on the white plate",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW5_put_the_blue_mug_on_the_white_plate_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw5_put_the_bread_on_oven_tray_and_push_it_in_the_oven=dict(
        lang="put the bread on oven tray and push it in the oven",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW5_put_the_bread_on_oven_tray_and_push_it_in_the_oven_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw5_put_the_bread_on_the_white_plate=dict(
        lang="put the bread on the white plate",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW5_put_the_bread_on_the_white_plate_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw5_put_the_yellow_bowl_on_oven_tray_and_push_it_in_the_oven=dict(
        lang="put the yellow bowl on oven tray and push it in the oven",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW5_put_the_yellow_bowl_on_oven_tray_and_push_it_in_the_oven_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw5_put_the_yellow_bowl_on_the_white_plate=dict(
        lang="put the yellow bowl on the white plate",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW5_put_the_yellow_bowl_on_the_white_plate_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw6_open_the_air_fryer_and_put_the_bowl_with_hot_dogs_in_it=dict(
        lang="open the air fryer and put the bowl with hot dogs in it",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW6_open_the_air_fryer_and_put_the_bowl_with_hot_dogs_in_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw6_open_the_air_fryer_and_put_the_bread_in_it=dict(
        lang="open the air fryer and put the bread in it",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW6_open_the_air_fryer_and_put_the_bread_in_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw6_open_the_air_fryer=dict(
        lang="open the air fryer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW6_open_the_air_fryer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw6_put_the_bowl_with_hot_dogs_on_the_white_plate=dict(
        lang="put the bowl with hot dogs on the white plate",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW6_put_the_bowl_with_hot_dogs_on_the_white_plate_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw6_put_the_bread_on_the_white_plate=dict(
        lang="put the bread on the white plate",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW6_put_the_bread_on_the_white_plate_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw6_put_the_pink_mug_on_the_white_plate=dict(
        lang="put the pink mug on the white plate",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW6_put_the_pink_mug_on_the_white_plate_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw7_open_the_air_fryer_and_put_the_blue_bowl_inside_it=dict(
        lang="open the air fryer and put the blue bowl inside it",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW7_open_the_air_fryer_and_put_the_blue_bowl_inside_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw7_open_the_fryer=dict(
        lang="open the fryer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW7_open_the_fryer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw7_put_the_blue_bowl_in_the_back_compartment_of_the_caddy=dict(
        lang="put the blue bowl in the back compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW7_put_the_blue_bowl_in_the_back_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw7_put_the_book_in_the_back_compartment_of_the_caddy=dict(
        lang="put the book in the back compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW7_put_the_book_in_the_back_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw7_put_the_book_in_the_front_compartment_of_the_caddy=dict(
        lang="put the book in the front compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW7_put_the_book_in_the_front_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw7_put_the_red_cup_in_the_front_compartment_of_the_caddy=dict(
        lang="put the red cup in the front compartment of the caddy",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW7_put_the_red_cup_in_the_front_compartment_of_the_caddy_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw8_open_the_bottom_drawer=dict(
        lang="open the bottom drawer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW8_open_the_bottom_drawer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw8_open_the_top_drawer_and_put_the_blue_mug_inside_it=dict(
        lang="open the top drawer and put the blue mug inside it",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW8_open_the_top_drawer_and_put_the_blue_mug_inside_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw8_open_the_top_drawer_and_put_the_pink_mug_inside_it=dict(
        lang="open the top drawer and put the pink mug inside it",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW8_open_the_top_drawer_and_put_the_pink_mug_inside_it_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw8_open_the_top_drawer=dict(
        lang="open the top drawer",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW8_open_the_top_drawer_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw8_put_the_blue_mug_on_the_white_plate=dict(
        lang="put the blue mug on the white plate",
        horizon=200,
        mutex_path=f"{mutex_hitl_path}/RW8_put_the_blue_mug_on_the_white_plate_demo.hdf5",
        mutex_filter_key=None,
    ),
    rw8_put_the_bread_on_the_white_plate=dict(
        lang="put the bread on the white plate",
        horizon=200,
        roudn01_path=f"{mutex_hitl_path}/RW8_put_the_bread_on_the_white_plate_demo.hdf5",
        mutex_filter_key=None,
    ),
)   

########################################

def get_ds_cfg(ds_names, exclude_ds_names=None, overwrite_ds_lang=False, src="human", filter_frac=None, eval=None, gen_tex=True):
    """ 
    ds_names defines the dataset group to be used
    src defines whether to use human or mg demos key (only relevant to robocasa) 
    """
    # ROBOCASA
    if ds_names == "robocasa_original": # can either be human demos, or mg >> defined by src
        dataset_group = ROBOCASA_ORIGINAL
    elif ds_names == "robocasa_round0":
        dataset_group = ROBOCASA_ORIGINAL
    elif ds_names == "robocasa_round01":  
        dataset_group = create_robocasa_hitl_tasks(robocasa_round01_path)
    elif ds_names == "robocasa_round012":
        dataset_group = create_robocasa_hitl_tasks(robocasa_round012_path)
    
    # MUTEX
    elif ds_names == "mutex_original": 
        dataset_group = create_mutex_hitl_tasks(mutex_demo_path)
    elif ds_names == "mutex_round0":
        dataset_group = create_mutex_hitl_tasks(mutex_demo_path)
    elif ds_names == "mutex_round01":
        dataset_group = create_mutex_hitl_tasks(mutex_round01_path)
    elif ds_names == "mutex_round012":
        dataset_group = create_mutex_hitl_tasks(mutex_round012_path)
    else:
        raise ValueError(f"Unknown dataset group: {ds_names}")
    
    """ 
    If training on the original demos, include all the tasks (for world model training)
    If training for hitl data, only include the hitl tasks
    """
    ds_names_lst = list(dataset_group.keys())
    if "robocasa_round" in ds_names:
        ds_names_lst = robocasa_hitl_tasks
    elif "mutex_round" in ds_names:
        ds_names_lst = mutex_hitl_tasks
    
    ret = []
    for name in ds_names_lst:
        ds_meta = dataset_group[name]

        cfg = dict(
            horizon=ds_meta["horizon"]
        )
        
        # determine whether we are performing eval on dataset
        if eval is None or name in eval:
            cfg["do_eval"] = True
        else:
            cfg["do_eval"] = False

        # if applicable overwrite the language stored in the dataset
        if overwrite_ds_lang is True:
            cfg["lang"] = ds_meta["lang"]
        
        # determine dataset path
        path_list = ds_meta.get(f"{src}_path", None)

        # skip if entry does not exist for this dataset src
        if path_list is None:
            continue
        
        if isinstance(path_list, str):
            path_list = [path_list]

        for path_i, path in enumerate(path_list):
            cfg_for_path = deepcopy(cfg)

            # determine dataset filter key
            if filter_frac is not None:
                total_demos = int(ds_meta[f"{src}_filter_key"].split("_demos")[0].split("_")[-1])
                num_filter_demos = int(total_demos * filter_frac)
                prefix = ds_meta[f"{src}_filter_key"].split(f"{total_demos}_demos")[0]
                cfg_for_path["filter_key"] = f"{prefix}{num_filter_demos}_demos"
            else:
                cfg_for_path["filter_key"] = ds_meta[f"{src}_filter_key"]

            if "env_meta_update_dict" in ds_meta:
                cfg_for_path["env_meta_update_dict"] = ds_meta["env_meta_update_dict"]
            
            if not path.endswith(".hdf5"):
                # determine path
                if gen_tex is True:
                    
                    if os.path.exists(os.path.join(path, "demo_gentex_im128_more.hdf5")):
                        path = os.path.join(path, "demo_gentex_im128_more.hdf5")
                    else:
                        path = os.path.join(path, "demo_gentex_im128.hdf5")
                        
                else:
                    path = os.path.join(path, "demo_im128.hdf5")
            cfg_for_path["path"] = path

            if path_i > 0:
                cfg_for_path["do_eval"] = False

            ret.append(cfg_for_path)

    return ret


def scan_datasets(folder, postfix=".h5"):
    dataset_paths = []
    for root, dirs, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            if f.endswith(postfix):
                dataset_paths.append(os.path.join(root, f))
    return dataset_paths
