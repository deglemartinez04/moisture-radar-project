# Recording protocol

Why this exists: with the recordings so far, every model plateaus at about 0.6–0.7 balanced accuracy on a new day.
The limit is the data: Damp has 5 rows from 2 boards, Dry comes from one day, every wet row was recorded at 50 mm
while dry spans 12–60 mm, and the labels come from how a board was wetted (the boards dried while being scanned)
because `reference_moisture_pct` was never filled in.

This protocol keeps the same kind of data (Wood + drywall, same app, same wetting methods) and changes how
sessions are run and labelled.

## Fixed for every recording
- **Stack:** Wood + drywall, drywall facing the sensor. Material = `Wood + drywall`.
- **Standoff:** sensor in its case at **50 mm**. Enter `50` in "Sensor standoff" every time. No other distances.
- **Settings:** click "Reset to project defaults" at the start of the day and never change settings between
  recordings or days. Background correction on every time. Tick "Save raw radar data to an .h5 file".
- **Length:** 300 frames per recording. Hold the sensor still.
- **Placements:** 3 per step. Lift the sensor off and set it down on a slightly different spot each time.
- **Check before recording:** the Amplitude Plot must show lobes beyond ~80 mm. If you only see the leakage spike
  near 60 mm and then a flat line (as in `Wood_Dry_4.h5`), re-seat the sensor and try again.

## Labels come from the pin meter
1. Read the pin meter at the sensor spot **before and after** each set of 3 placements.
2. Enter the mean in the app's `reference_moisture_pct` field.
3. The label follows the team cutoffs in `ml/ml/labels.py`:

   | Pin meter (% MC) | Label |
   |---|---|
   | below 12 | Dry |
   | 12 to 20 | Damp |
   | above 20 | Saturated |

   The label you choose in the app must match the number.
4. **Boundary steps:** if the before and after readings land in different classes, or the mean is within 1 point
   of a cutoff (11–13 or 19–21), write `boundary` on the sheet. Keep the data, but use it only as test data.
5. **Never relabel by hand afterwards.** If a label turns out wrong, put a line in a label-override file and
   write why (as V14 did).

## Boards and days
- Give each board its own `sample_id` (`B01`, `B02`, ...) and keep it for the whole project.
- Record **all three classes on every day**, so the day never lines up with the class.
- Rotate boards between days, so each class has different boards on different days.
- Dry data comes from at least 3 days and 2 boards, all at 50 mm.
- Reach each class with the wetting methods already used (pour 160 mL for Damp, soaks for Saturated), then
  **trust the pin meter, not the method**. Adjust the soak time until the reading is where you want it.

## A day (about 3 hours)
1. **0:00** Fill in the top of the session sheet. Power up, wait for the temperature to settle, reset to defaults.
2. **0:15 Dry board:** pin reading, 3 placements, pin reading.
3. **Wet boards:** wet each board, then scan a step about every 20 min for 1–2 hours. Each step is: pin reading,
   3 placements, pin reading, and one line on the sheet. The board drying out gives you Saturated, Damp and
   near-Dry steps on the same board, and that is useful as long as every step has its own pin reading.
4. **End of day:** copy `data_log.csv`, every `.h5` and `_background.npz`, and the session sheet into
   `data/recordings/<date>/`. Do not leave the `.h5` files only on the Pi.
5. Run `python ml/check_session.py data/recordings/<date>/data_log.csv` and fix anything it flags while the
   boards are still there.

Session sheet: copy `data/templates/session_log_template.csv`. One row per step, for example:
`2026-11-10,B03,4,soak_1h,10:05,11:20,75,0,22.4,19.6,21.0,Saturated,n,22,45,`

## Targets before retraining
- At least **30 recordings per class**, from at least **3 boards** and **3 days**.
- Keep the **last day completely out** of training and model choice. It is the final test.
- Then rerun `ml/compare_models.py` and `ml/train_subset.py` and compare with the current numbers.

## Optional
A wood-alone scan at each step gives a cross-check on the labels (the wood echo ratio, as in the 10-29 and 10-31
sessions). With pin readings it is not required.
