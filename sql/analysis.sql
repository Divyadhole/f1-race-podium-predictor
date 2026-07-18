-- Load data/processed/driver_race_features.csv as driver_race_features.

-- Constructor performance and podium conversion.
SELECT
    constructor_name,
    COUNT(*) AS driver_starts,
    SUM(podium) AS podiums,
    ROUND(AVG(CAST(podium AS DECIMAL)), 4) AS podium_rate,
    ROUND(AVG(points), 2) AS points_per_driver_start
FROM driver_race_features
GROUP BY constructor_name
ORDER BY podium_rate DESC;

-- Grid conversion performance.
SELECT
    grid,
    COUNT(*) AS starts,
    SUM(podium) AS podiums,
    ROUND(AVG(CAST(podium AS DECIMAL)), 4) AS podium_rate,
    ROUND(AVG(position), 2) AS average_finish
FROM driver_race_features
GROUP BY grid
ORDER BY grid;

-- Biggest drives through the field.
SELECT
    date,
    name AS race_name,
    driver_name,
    constructor_name,
    grid,
    position,
    grid - position AS positions_gained
FROM driver_race_features
WHERE status = 'Finished'
ORDER BY positions_gained DESC, date DESC
LIMIT 20;

