# InBody Tracking Feature

## Overview
The InBody tab provides comprehensive body composition tracking and analysis with automated metric calculations and visualizations.

## Data Schema
Each InBody measurement includes:

### Core Metrics
- `date` - Measurement date (YYYY-MM-DD)
- `inbody_score` - Overall InBody score
- `weight_kg` - Total body weight in kg
- `skeletal_muscle_kg_total` - Total skeletal muscle mass in kg
- `body_fat_kg_total` - Total body fat mass in kg
- `body_fat_percent` - Body fat percentage
- `visceral_fat_level` - Visceral fat level (1-20)
- `bmr_kcal` - Basal metabolic rate in kcal/day

### Segmental Muscle (kg)
- `muscle_right_arm_kg`
- `muscle_left_arm_kg`
- `muscle_trunk_kg`
- `muscle_right_leg_kg`
- `muscle_left_leg_kg`

### Segmental Fat (kg)
- `fat_right_arm_kg`
- `fat_left_arm_kg`
- `fat_trunk_kg`
- `fat_right_leg_kg`
- `fat_left_leg_kg`

### Notes
- `notes` - Free-text comments

## Derived Metrics
The app automatically calculates:

1. **Arm Asymmetry** - `arm_asym_pct = ((right_arm - left_arm) / avg) × 100`
2. **Leg Asymmetry** - `leg_asym_pct = ((right_leg - left_leg) / avg) × 100`
3. **Trunk Muscle Share** - `trunk_muscle_share_pct = (trunk / skeletal_muscle_total) × 100`
4. **Upper/Lower Ratio** - `upper_lower_ratio = (arms + trunk) / (legs)`
5. **Trunk to Limb Fat Ratio** - `trunk_to_limb_fat_ratio = trunk_fat / limb_fat`
6. **Monthly Deltas** - Changes in skeletal muscle, body fat, and weight

## Visualizations

### 1. KPI Summary
- Latest InBody score
- Δ Skeletal Muscle (with color-coded trend)
- Δ Body Fat (with color-coded trend)

### 2. Composition Trends
Line chart showing:
- Skeletal muscle mass (kg)
- Body fat mass (kg)
- Body fat percentage (%)

### 3. Visceral Fat Trend
Area chart tracking visceral fat level over time

### 4. Segmental Muscle Chart
Bar chart of muscle distribution across body segments (latest measurement)

### 5. Segmental Fat Chart
Bar chart of fat distribution across body segments (latest measurement)

### 6. Asymmetry Analysis
Line chart with ±5% tolerance bands showing:
- Arm muscle asymmetry
- Leg muscle asymmetry

### 7. Data Table
Comprehensive table with all measurements and computed metrics

## Sample Data
A demo CSV is provided at `data/sample_inbody.csv` with 3 measurements for reference. You can use these values as a guide when entering your first InBody measurements manually.

## Usage

### Adding Data
1. Navigate to the "Add / Edit" tab
2. Find the "🔬 InBody" card
3. Click "Add New" sub-tab
4. Fill in all measurement fields:
   - **Date**: Select measurement date
   - **Basic metrics**: InBody Score, Weight
   - **Body Composition**: Skeletal Muscle, Body Fat (kg & %), Visceral Fat, BMR
   - **Segmental Muscle**: Values for Right Arm, Left Arm, Trunk, Right Leg, Left Leg
   - **Segmental Fat**: Values for Right Arm, Left Arm, Trunk, Right Leg, Left Leg
   - **Notes**: Optional free-text comments
5. Click "Add InBody" to save

### Editing Data
1. In the "Edit" sub-tab, select an entry from the dropdown
2. Click "Load" to populate the form
3. Modify any fields
4. Click "Save" to update or "Delete" to remove

### Viewing Results
1. Navigate to the "🔬 InBody" tab
2. Charts and metrics update automatically
3. View comprehensive analysis across all visualizations

### Data Safety
- All numeric columns are safely coerced with error handling
- Empty or missing data doesn't crash charts
- Division by zero is handled in derived metrics

## Integration
The InBody feature is fully integrated with the existing app:
- Uses the same repository pattern (SQLite/Combined)
- Follows the same naming conventions
- Styled consistently with other tabs
- No modifications to existing Lifts, Bodyweight, or Measurements features

## Technical Details
- **UI**: `gymtracker/ui.py` - InBody nav_panel with 6 chart cards
- **Server**: `gymtracker/server.py` - `inbody_df()` reactive calc + 7 render functions
- **Schema**: `gymtracker/utils.py` - REQUIRED_TABS["InBody"]
- **Database**: `gymtracker/repos.py` - InBody table creation and migration support
- **Sample Data**: No demo CSV is included in this repository; import your own CSV or enter measurements manually.

## Charts Technology
- Uses Plotly via shinywidgets for interactive visualizations
- Consistent 520px height for main charts, 400px for secondary
- Color-coded for easy interpretation
- Responsive tooltips with formatted values
