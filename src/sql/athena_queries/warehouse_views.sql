-- ============================================================
-- IMDb Athena Warehouse Views
-- These replace Redshift for analytics on student AWS accounts
-- ============================================================

-- View 1: Decade stats
CREATE OR REPLACE VIEW imdb_analytics.v_decade_stats AS
SELECT
    decade,
    COUNT(*)                        AS movie_count,
    ROUND(AVG(averagerating), 2)   AS avg_rating,
    MAX(averagerating)             AS max_rating,
    SUM(numvotes)                  AS total_votes,
    SUM(is_high_rated)             AS high_rated_count
FROM imdb_analytics.cleaned
GROUP BY decade
ORDER BY decade;

-- View 2: Genre stats
CREATE OR REPLACE VIEW imdb_analytics.v_genre_stats AS
SELECT
    genres,
    COUNT(*)                        AS movie_count,
    ROUND(AVG(averagerating), 2)   AS avg_rating,
    SUM(numvotes)                  AS total_votes,
    SUM(is_high_rated)             AS high_rated_count
FROM imdb_analytics.cleaned
WHERE numvotes >= 1000
GROUP BY genres
HAVING COUNT(*) >= 50
ORDER BY avg_rating DESC;

-- View 3: Full movie summary
CREATE OR REPLACE VIEW imdb_analytics.v_movies_summary AS
SELECT
    tconst,
    primarytitle,
    startyear,
    decade,
    genres,
    averagerating,
    numvotes,
    rating_tier,
    is_high_rated,
    runtime_bucket,
    vote_tier
FROM imdb_analytics.cleaned;

-- View 4: Top movies
CREATE OR REPLACE VIEW imdb_analytics.v_top_movies AS
SELECT
    primarytitle,
    startyear,
    genres,
    averagerating,
    numvotes,
    rating_tier,
    runtime_bucket
FROM imdb_analytics.cleaned
WHERE numvotes >= 50000
ORDER BY averagerating DESC;
