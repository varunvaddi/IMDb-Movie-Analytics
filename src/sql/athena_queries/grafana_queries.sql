-- ============================================================
-- Grafana Dashboard Queries
-- ============================================================

-- Dashboard 1: Movie Analytics

-- Panel: Total movies
SELECT COUNT(*) AS total_movies
FROM imdb_analytics.cleaned;

-- Panel: Average rating
SELECT ROUND(AVG(averagerating), 2) AS avg_rating
FROM imdb_analytics.cleaned;

-- Panel: Total votes
SELECT SUM(numvotes) AS total_votes
FROM imdb_analytics.cleaned;

-- Panel: High rated count
SELECT SUM(is_high_rated) AS high_rated
FROM imdb_analytics.cleaned;

-- Panel: Movies per decade
SELECT decade, COUNT(*) AS movie_count
FROM imdb_analytics.cleaned
WHERE decade IS NOT NULL
GROUP BY decade
ORDER BY decade;

-- Panel: Avg rating by decade
SELECT decade, ROUND(AVG(averagerating), 2) AS avg_rating
FROM imdb_analytics.cleaned
WHERE decade IS NOT NULL
GROUP BY decade
ORDER BY decade;

-- Panel: Top 15 genres by rating
SELECT genres,
    COUNT(*) AS movie_count,
    ROUND(AVG(averagerating), 2) AS avg_rating
FROM imdb_analytics.cleaned
WHERE numvotes >= 1000
GROUP BY genres
HAVING COUNT(*) >= 20
ORDER BY avg_rating DESC
LIMIT 15;

-- Panel: Rating tier breakdown
SELECT rating_tier, COUNT(*) AS movie_count
FROM imdb_analytics.cleaned
GROUP BY rating_tier
ORDER BY movie_count DESC;

-- Panel: Vote tier breakdown
SELECT vote_tier, COUNT(*) AS movie_count
FROM imdb_analytics.cleaned
GROUP BY vote_tier
ORDER BY movie_count DESC;

-- Dashboard 3: Pipeline Monitoring

-- Panel: Data quality
SELECT
    COUNT(*) AS total_movies,
    ROUND(COUNT(primarytitle) * 100.0 / COUNT(*), 2) AS title_completeness,
    ROUND(COUNT(averagerating) * 100.0 / COUNT(*), 2) AS rating_completeness,
    ROUND(COUNT(numvotes) * 100.0 / COUNT(*), 2) AS votes_completeness
FROM imdb_analytics.cleaned;
