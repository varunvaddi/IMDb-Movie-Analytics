-- Grafana CloudWatch + Athena queries for monitoring dashboards

-- Pipeline health check
SELECT
    COUNT(*) AS total_records,
    ROUND(AVG(averagerating), 3) AS mean_rating,
    COUNT(DISTINCT decade) AS decades_covered,
    SUM(is_high_rated) AS high_rated_count,
    ROUND(COUNT(primarytitle) * 100.0 / COUNT(*), 2) AS data_completeness_pct
FROM imdb_analytics.cleaned;

-- Data quality by decade
SELECT
    decade,
    COUNT(*) AS records,
    ROUND(AVG(averagerating), 2) AS avg_rating,
    COUNT(runtimeminutes) AS has_runtime,
    ROUND(COUNT(runtimeminutes) * 100.0 / COUNT(*), 1) AS runtime_completeness
FROM imdb_analytics.cleaned
WHERE decade IS NOT NULL
GROUP BY decade
ORDER BY decade;
