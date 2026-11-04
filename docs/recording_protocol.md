# Recording protocol

Why this exists: with the recordings so far, every model plateaus at about 0.6–0.7 balanced accuracy on a new day.
The limit is the data: only 4 wood studs (S1–S4) have been tested, all Dry data comes from one stud (S1), every wet
row was recorded at 50 mm while dry spans 12–60 mm, and the labels come from how a stud was wetted (the studs dried
while being scanned).

This protocol keeps the same kind of data (Wood + drywall, same app, same wetting methods) and changes how
sessions are run and labelled. **The team does not take pin-meter readings, so labels come from the wetting
procedure and the time since wetting.** That is the weakest part of the data (see "Known limitation").

## Fixed for every recording
- **Stack:** Wood + drywall, drywall facing the sensor. Material = `Wood + drywall`.
- **Standoff:** sensor in its case at **50 mm**. Enter `50` in "Sensor standoff" every time. No other distances.
- **Settings:** click "Reset to project defaults" at the start of the day and never change settings between
  recordings or days. Background correction on every time. Tick "Save raw radar data to an .h5 file".
- **Length:** 300 frames per recording. Hold the sensor still.
- **Placements:** 3 per step. Lift the sensor off and set it down on a slightly different spot each time.
- **Check before recording:** the Amplitude Plot must show lobes beyond ~80 mm. If you only see the leakage spike
  near 60 mm and then a flat line (as in `Wood_Dry_4.h5`), re-seat the sensor and try again.

## Labels come from the wetting procedure and time
1. **Write down the label rules before the first session and don't change them afterwards.** For each wetting
   method (for example `pour_160mL`, `soak_1h`, `soak_2h`), decide which `time_since_wet_min` range counts as
   Saturated, Damp and Dry. Save the table next to the session sheet.
2. **Record the procedure on every scan.** In the app, fill in `sample_id` (the stud), `water_amount_ml` and
   `time_since_wet_min`, and name the method on the session sheet. For soaks, `water_amount_ml` can stay 0, but
   `time_since_wet_min` must be filled in.
3. **Label each recording from the table**, using the time of that scan. A stud soaked for 2 hours is not
   Saturated for the whole afternoon, so one label for a whole session is wrong.
4. **Take a wood-alone scan at each step** (the stud without drywall, same standoff). It is the only measurement
   we have of how wet the wood really is. V14 relabelled the drying series from its echo, comparing the front
   echo with the dry-wood echo at 50 mm (1.75 or more = Saturated, 1.15–1.75 = Damp). Those thresholds were set by
   hand, so recheck them against new dry baselines.
5. **Never relabel by hand afterwards.** If a label turns out wrong, put a line in a label-override file and
   write why (as V14 did).

**Known limitation:** the labels are not measured. A stud's real moisture can differ from what its procedure and
time suggest, so some labels will be wrong, and no model can be more accurate than its labels. Say so when
reporting results. If a moisture meter or scale becomes available later, add its readings; the checker will use them.

## Studs and days
- Each wood stud has its own `sample_id`. S1–S4 are taken; new studs continue as `S5`, `S6`, ... Keep a stud's ID for the whole project.
- Record **all three classes on every day**, so the day never lines up with the class.
- Rotate studs between days, so each class has different studs on different days.
- **Every new session includes at least one stud the model has never seen** (a new `sample_id`). That is the test of whether the model works on a new stud and not only on S1–S4.
- Dry data comes from at least 3 days and 2 studs, all at 50 mm.
- Reach each class with the wetting methods already used (pour 160 mL for Damp, soaks for Saturated).

## A day (about 3 hours)
1. **0:00** Fill in the top of the session sheet. Power up, wait for the temperature to settle, reset to defaults.
2. **0:15 Dry stud:** 3 placements, one line on the sheet.
3. **Wet studs:** wet each stud and note the start time, then scan a step about every 20 min for 1–2 hours. Each
   step is: 3 placements (with drywall), 3 placements (wood alone), and one line on the sheet. The stud drying
   out gives you Saturated, Damp and near-Dry steps on the same stud, and that is useful as long as every step has
   its own `time_since_wet_min` and its own label.
4. **End of day:** copy `data_log.csv`, every `.h5` and `_background.npz`, and the session sheet into
   `data/recordings/<date>/`. Do not leave the `.h5` files only on the Pi.
5. Run `python ml/check_session.py data/recordings/<date>/data_log.csv` and fix anything it flags while the
   studs are still there.

Session sheet: copy `data/templates/session_log_template.csv`. One row per step, for example:
`2026-11-10,S5,4,soak_1h,10:05,11:20,75,0,y,Damp,22,45,`

## Targets before retraining
- At least **30 recordings per class**, from at least **3 studs** and **3 days**.
- Keep the **last day completely out** of training and model choice. It is the final test.
- Then rerun `ml/compare_models.py` and `ml/train_subset.py` and compare with the current numbers.
