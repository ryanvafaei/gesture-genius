"""
Think: the garden. It grows from showing up, not from performance.

  * every session with at least one completed exercise waters it, also on
    difficult days; nothing ever wilts, missing days changes nothing;
  * each watering moves the current plant one stage on:
    seed -> sprout -> leaves -> bud -> flower (GARDEN_STAGES);
  * a flower means the next watering plants a new seed in the next plot,
    of a type she picks from two; a full bed begins "a new season";
  * a personal best adds a bee, a milestone a butterfly. Extras are only
    ever added.

Pure state changes on the garden dict (storage.new_garden()); drawing is in
rehab/ui.py.
"""

from rehab import config
from rehab.events import event


def current(garden):
    return garden["plots"][-1] if garden["plots"] else None


def needs_new_plant(garden):
    plot = current(garden)
    return plot is None or plot["stage"] >= config.GARDEN_STAGES - 1


def plant_options(garden, types=config.PLANT_TYPES):
    """The two plant types offered for the next seed (varies from bed to bed)."""
    i = (garden.get("waterings", 0) // config.GARDEN_STAGES) % len(types)
    return types[i], types[(i + 1) % len(types)]


def water(garden, plant=None, bee=False, butterflies=0):
    """Grow the garden once. Returns the GardenGrew event."""
    new_season = False
    new_plant = needs_new_plant(garden)
    if new_plant:
        if len(garden["plots"]) >= config.GARDEN_PLOTS:
            garden["season"] = garden.get("season", 1) + 1
            garden["plots"] = []
            new_season = True
        plant = plant or plant_options(garden)[0]
        garden["plots"].append({"plant": plant, "stage": 0})
        prev_stage = None
    else:
        plot = current(garden)
        prev_stage = plot["stage"]
        plot["stage"] += 1
    garden["waterings"] = garden.get("waterings", 0) + 1
    if bee:
        garden["bees"] = garden.get("bees", 0) + 1
    garden["butterflies"] = garden.get("butterflies", 0) + butterflies
    plot = current(garden)
    return event("GardenGrew", plant=plot["plant"], stage=plot["stage"], prev_stage=prev_stage,
                 plot=len(garden["plots"]) - 1, new_plant=new_plant, new_season=new_season,
                 bee=bool(bee), butterflies=butterflies)
