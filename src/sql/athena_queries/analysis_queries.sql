-- ============================================================
-- IMDb Analytics Queries
-- Run these in AWS Athena console or via CLI
-- ============================================================

-- Query 1: How many movies per decade?
SELECT
    decade,
    COUNT(*) as movie_count,
    ROUND(AVG(averagerating), 2) as avg_rating,
    MAX(numvotes) as max_votes
FROM imdb_analytics.cleaned
GROUP BY decade
ORDER BY decade;

-- Query 2: Rating tier breakdown
SELECT
    rating_tier,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
FROM imdb_analytics.cleaned
GROUP BY rating_tier
ORDER BY count DESC;

-- Query 3: Top 20 highest rated movies (min 50k votes for credibility)
SELECT
    primarytitle,
    startyear,
    averagerating,
    numvotes,
    genres,
    runtime_bucket
FROM imdb_analytics.cleaned
WHERE numvotes >= 50000
ORDER BY averagerating DESC
LIMIT 20;

-- Query 4: Best genres by average rating
SELECT
    genres,
    COUNT(*) as movie_count,
    ROUND(AVG(averagerating), 2) as avg_rating,
    SUM(numvotes) as total_votes
FROM imdb_analytics.cleaned
WHERE numvotes >= 1000
GROUP BY genres
HAVING COUNT(*) >= 50
ORDER BY avg_rating DESC
LIMIT 20;

-- Query 5: Rating trends over decades
SELECT
    decade,
    rating_tier,
    COUNT(*) as count
FROM imdb_analytics.cleaned
WHERE decade >= 1960
GROUP BY decade, rating_tier
ORDER BY decade, rating_tier;

-- Query 6: Does runtime affect rating?
SELECT
    runtime_bucket,
    COUNT(*) as count,
    ROUND(AVG(averagerating), 2) as avg_rating,
    ROUND(AVG(numvotes), 0) as avg_votes
FROM imdb_analytics.cleaned
WHERE runtime_bucket != 'unknown'
GROUP BY runtime_bucket
ORDER BY avg_rating DESC;
