# Disclaimer & Fair Play Notice

## Is this a Cheat?
**No.** The Adaptive Filter is a **latency compensation and smoothing tool**, similar to "Monitor Overdrive" or high-polling-rate hardware. It is designed to make your tablet feel more responsive by predicting your hand's physical movement.

## How it Works
The filter uses a lightweight Neural Network to predict your future cursor position based **only** on your recent movement history (last ~40ms).
*   **Input:** Tablet X/Y coordinates (Physics)
*   **Output:** Predicted X/Y coordinates (Physics)

## Technical Limitations (Why it's not an Aimbot)
1.  **No Game Access:** The plugin runs entirely within OpenTabletDriver. It has **zero access** to the osu! process, memory, map files, or audio timing. It does not know where hit circles are.
2.  **Short Memory:** The network only "remembers" the last 5-6 input points. It cannot "replay" a map because it cannot remember the start of a pattern.
3.  **Ambiguity:** The filter cannot memorize map patterns because the same screen coordinate is used for different movements across different maps.

## Theoretical Edge Cases
While the filter is designed for physics prediction, users should be aware of theoretical behaviors when using maximum settings:

### "Overfitting" to Hand Physics
The model *will* learn your specific biomechanics (e.g., "User tends to reverse direction quickly"). This is intended behavior and is what makes the filter "Adaptive." It is learning *you*, not the game.

### Absolute Position & "Ranked Slop"
If **"Use Absolute Position"** is enabled and you play highly repetitive maps (e.g., "1-2 jumps" centered on the screen) for extended periods:
*   The model *may* develop a bias towards the center of the screen or common flow directions.
*   This is not "knowing" where the circle is, but rather learning a statistical probability (e.g., "If at edge, likely to move Center").
*   **Recommendation:** If you are concerned about the ethical implications of this specific bias, you can disable **"Use Absolute Position"** in the settings. This forces the model to rely purely on velocity/acceleration (Deltas), which is 100% ethically neutral.

## Usage in Tournaments
Always check with tournament organizers before using custom drivers or plugins. While this plugin does not interact with the game client, some tournaments have strict rules regarding "predictive" or "smoothing" software.
