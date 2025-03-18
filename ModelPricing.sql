--------- 1. Data preparation

------ 1.1 Data cleaning

SELECT * FROM public.sales_data

--- Identify duplicates

SELECT product_merchant_identifier, deal_number, COUNT(*) AS count
FROM sales_data
GROUP BY product_merchant_identifier, deal_number
HAVING COUNT(*) > 1;

-- the product 3ZBXWEA9 shows duplicates

SELECT *
FROM sales_data
WHERE product_merchant_identifier = '3ZBXWEA9';

--- Detect null values
SELECT 
    COUNT(CASE WHEN product_merchant_identifier IS NULL THEN 1 END) AS null_pm_identifier,
    COUNT(CASE WHEN "name" IS NULL THEN 1 END) AS null_name,
    COUNT(CASE WHEN deal_number IS NULL THEN 1 END) AS null_deal_number,
	COUNT(CASE WHEN deal_sum IS NULL THEN 1 END) AS null_deal_sum,
    COUNT(CASE WHEN brand_name IS NULL THEN 1 END) AS null_brand_name,
    COUNT(CASE WHEN category_name IS NULL THEN 1 END) AS null_category_name,
	COUNT(CASE WHEN assortment_type IS NULL THEN 1 END) AS null_assortment_type,
    COUNT(CASE WHEN earliest_delivery_date IS NULL THEN 1 END) AS null_ed_date,
    COUNT(CASE WHEN delivered_quantity IS NULL THEN 1 END) AS null_delivered_quantity,
	COUNT(CASE WHEN gross_sales IS NULL THEN 1 END) AS gross_sales,
    COUNT(CASE WHEN net_sales_predicted IS NULL THEN 1 END) AS null_net_sales_predicted,
    COUNT(CASE WHEN net_revenue_incl_service_fees_excl_vat_predicted IS NULL THEN 1 END) AS null_netr,
	COUNT(CASE WHEN fulfillment_costs_predicted IS NULL THEN 1 END) AS null_costp
FROM sales_data;

SELECT product_merchant_identifier
FROM sales_data
WHERE brand_name IS NULL;

SELECT product_merchant_identifier, brand_name
FROM reduction_report
WHERE product_merchant_identifier = 'AVVF8U2M' OR product_merchant_identifier = 'ZPRWVQWS' 
OR product_merchant_identifier = '3KK2DF1R' OR product_merchant_identifier = 'NZ97TJSS'
OR product_merchant_identifier = 'QKEDRZ0E' OR product_merchant_identifier = '07C1W1KN'
OR product_merchant_identifier = 'VX8YGH0A' OR product_merchant_identifier = 'B32DT5IF'
OR product_merchant_identifier = 'VJ0X7KZ2' OR product_merchant_identifier = 'D4Z5TRNS'
OR product_merchant_identifier = 'YTJDVOHH' OR product_merchant_identifier = '5BXB7X4G'
OR product_merchant_identifier = '6Y2EJPQJ' OR product_merchant_identifier = 'M8HACSBN';

--- Unify brand names and categories

SELECT DISTINCT brand_name
FROM sales_data
ORDER BY brand_name ASC;

SELECT DISTINCT category_name
FROM sales_data
ORDER BY category_name ASC;

--- Creat a process

CREATE PROCEDURE clean_data()
BEGIN
    -- detect duplicates
    SELECT product_merchant_identifier, deal_number, COUNT(*) AS count
    FROM sales_data
    GROUP BY product_merchant_identifier, deal_number
    HAVING COUNT(*) > 1;

    -- detect null values
    SELECT 
    COUNT(CASE WHEN product_merchant_identifier IS NULL THEN 1 END) AS null_pm_identifier,
    COUNT(CASE WHEN "name" IS NULL THEN 1 END) AS null_name,
    COUNT(CASE WHEN deal_number IS NULL THEN 1 END) AS null_deal_number,
	COUNT(CASE WHEN deal_sum IS NULL THEN 1 END) AS null_deal_sum,
    COUNT(CASE WHEN brand_name IS NULL THEN 1 END) AS null_brand_name,
    COUNT(CASE WHEN category_name IS NULL THEN 1 END) AS null_category_name,
	COUNT(CASE WHEN assortment_type IS NULL THEN 1 END) AS null_assortment_type,
    COUNT(CASE WHEN earliest_delivery_date IS NULL THEN 1 END) AS null_ed_date,
    COUNT(CASE WHEN delivered_quantity IS NULL THEN 1 END) AS null_delivered_quantity,
	COUNT(CASE WHEN gross_sales IS NULL THEN 1 END) AS gross_sales,
    COUNT(CASE WHEN net_sales_predicted IS NULL THEN 1 END) AS null_net_sales_predicted,
    COUNT(CASE WHEN net_revenue_incl_service_fees_excl_vat_predicted IS NULL THEN 1 END) AS null_netr,
	COUNT(CASE WHEN fulfillment_costs_predicted IS NULL THEN 1 END) AS null_costp
	FROM sales_data;

    -- Unify names
    SELECT DISTINCT brand_name
    FROM sales_data
    ORDER BY brand_name ASC;

    SELECT DISTINCT category_name
    FROM sales_data
    ORDER BY category_name ASC;
END;

SELECT * FROM public.reduction_report

-- Identify duplicates

SELECT product_merchant_identifier, COUNT(*) AS count
FROM reduction_report
GROUP BY product_merchant_identifier
HAVING COUNT(*) > 1;

-- detect null values
SELECT 
    COUNT(CASE WHEN product_merchant_identifier IS NULL THEN 1 END) AS null_pm_identifier,
    COUNT(CASE WHEN assortment_type IS NULL THEN 1 END) AS null_assortype,
    COUNT(CASE WHEN brand_name IS NULL THEN 1 END) AS null_brand_name,
    COUNT(CASE WHEN season_class IS NULL THEN 1 END) AS null_season_class,
	COUNT(CASE WHEN performance_score_ayo IS NULL THEN 1 END) AS null_perf_score_ayo,
    COUNT(CASE WHEN price_before_reductions IS NULL THEN 1 END) AS null_price_berfore_reduct,
    COUNT(CASE WHEN sale_discount IS NULL THEN 1 END) AS null_sale_discount,
	COUNT(CASE WHEN sale_price IS NULL THEN 1 END) AS null_sale_price,
    COUNT(CASE WHEN total_costs IS NULL THEN 1 END) AS null_total_costs,
    COUNT(CASE WHEN return_rate IS NULL THEN 1 END) AS null_return_rate,
	COUNT(CASE WHEN service_fees IS NULL THEN 1 END) AS null_service_fees,
	COUNT(CASE WHEN vat_percentage IS NULL THEN 1 END) AS null_vat_percentage,
	COUNT(CASE WHEN max_reduction_outlet IS NULL THEN 1 END) AS null_max_reduction_outlet

FROM reduction_report;

-- Unify names

SELECT DISTINCT brand_name
FROM sales_data
ORDER BY brand_name ASC;

SELECT DISTINCT category_name
FROM sales_data
ORDER BY category_name ASC;


------ 1.2 Data transfomation

-- compute STR, performance_class, pc_target_adjustment, PC_target

ALTER TABLE sales_data
ADD COLUMN STR DECIMAL,
ADD COLUMN performance_class VARCHAR(50),
ADD COLUMN pc_target_adjustment INT,
ADD COLUMN PC_target DECIMAL;

--ALTER TABLE sales_data
--DROP COLUMN PC_target;


UPDATE sales_data
SET

-- Compute STR
    STR = (net_sales_predicted::decimal / delivered_quantity::decimal),
	
   -- Assign performance category based on STR
    performance_class = CASE 
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) = 0 THEN 'Low'
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) > 0 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.10 THEN 'Medium-low'
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.10 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.25 THEN 'Medium-high'
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.25 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.40 THEN 'High'
        ELSE 'Unknown'
    END,
	
    -- Adjust pp based on  performance category and deals numbers
    pc_target_adjustment = CASE
        -- Low
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) = 0 AND deal_number = 1 THEN -2
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) = 0 AND deal_number = 2 THEN -2
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) = 0 AND deal_number = 3 THEN -3
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) = 0 AND deal_number = 4 THEN -3
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) = 0 AND deal_number >= 5 THEN -4

        -- Medium-low
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) > 0 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.10 AND deal_number = 1 THEN -2
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) > 0 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.10 AND deal_number = 2 THEN -2
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) > 0 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.10 AND deal_number = 3 THEN -3
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) > 0 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.10 AND deal_number = 4 THEN -3
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) > 0 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.10 AND deal_number >= 5 THEN -4

        -- Medium-high
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.10 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.25 AND deal_number = 1 THEN -1
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.10 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.25 AND deal_number = 2 THEN -2
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.10 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.25 AND deal_number = 3 THEN -2
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.10 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.25 AND deal_number = 4 THEN -3
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.10 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.25 AND deal_number >= 5 THEN -3

        -- High
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.25 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.40 AND deal_number = 1 THEN 0
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.25 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.40 AND deal_number = 2 THEN 0
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.25 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.40 AND deal_number = 3 THEN 0
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.25 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.40 AND deal_number = 4 THEN -1
        WHEN (net_sales_predicted::decimal / delivered_quantity::decimal) >= 0.25 
             AND (net_sales_predicted::decimal / delivered_quantity::decimal) < 0.40 AND deal_number >= 5 THEN -1
        
        ELSE 0
    END,


UPDATE sales_data
SET
    -- Compute PC_target
    PC_target = 30 + PC_target_adjustment


----- check null vales of variables created

SELECT 
    COUNT(CASE WHEN str IS NULL THEN 1 END) AS null_STR,
    COUNT(CASE WHEN performance_class IS NULL THEN 1 END) AS null_performance_class,
    COUNT(CASE WHEN pc_target_adjustment IS NULL THEN 1 END) AS null_pc_target_adjustment,
    COUNT(CASE WHEN PC_target IS NULL THEN 1 END) AS null_PC_target

FROM sales_data;

SELECT product_merchant_identifier, STR, net_sales_predicted, delivered_quantity, performance_class, 
pc_target_adjustment, pc_target
FROM sales_data
WHERE performance_class = 'Unknown'

------ Compute final reduction

ALTER TABLE reduction_report
ADD COLUMN final_reduction FLOAT;

--ALTER TABLE reduction_report
--DROP COLUMN final_reduction;

WITH higher_gross AS (
    -- Select the highiest gross
    SELECT 
        s.product_merchant_identifier,
        MAX(s.gross_sales) AS higher_gross_number
    FROM 
        sales_data s
    GROUP BY 
        s.product_merchant_identifier
),
higher_gross_data AS (
    -- Unir con los datos del último deal para obtener el pc_target y otros datos
    SELECT 
        l.product_merchant_identifier,
        s.pc_target / 100.0 AS pc_target,  -- Convertimos el PC_target a porcentaje
        r.total_costs,
        r.return_rate,
        r.service_fees,
        r.vat_percentage,
        r.sale_price
    FROM 
        higher_gross l
    JOIN 
        sales_data s
    ON 
        l.product_merchant_identifier = s.product_merchant_identifier
        AND l.higher_gross_number = s.gross_sales
    JOIN 
        reduction_report r
    ON 
        l.product_merchant_identifier = r.product_merchant_identifier
)
-- Actualizar la tabla reduction_report_pr con la reducción final calculada
UPDATE reduction_report r
SET final_reduction = (
    1 - (
        (1 + higher_gross_data.vat_percentage) * higher_gross_data.total_costs / 
        ((1 - higher_gross_data.pc_target) * (1 - higher_gross_data.return_rate) * (1 + higher_gross_data.service_fees) * higher_gross_data.sale_price)
    )
)
FROM higher_gross_data
WHERE r.product_merchant_identifier = higher_gross_data.product_merchant_identifier;


SELECT *
FROM reduction_report
WHERE final_reduction > 1;

SELECT 
    COUNT(CASE WHEN final_reduction IS NULL THEN 1 END) AS null_final_reduction
FROM reduction_report; --- 937

SELECT COUNT(DISTINCT product_merchant_identifier)
FROM sales_data;---9027


SELECT COUNT(DISTINCT product_merchant_identifier)
FROM reduction_report;--- 9964

------ diff. = 937
SELECT *
FROM reduction_report
WHERE product_merchant_identifier IN (
    SELECT DISTINCT product_merchant_identifier
    FROM sales_data
);

----------- Data analysis

CREATE TABLE DA_insights AS
SELECT *
FROM reduction_report;

DELETE FROM da_insights
WHERE final_reduction IS NULL; -- delete nulls

--- bring deal_number

ALTER TABLE da_insights ADD COLUMN deal_number INTEGER;

WITH max_sales_deal AS (
    SELECT 
        product_merchant_identifier,
        deal_number
    FROM (
        SELECT 
            product_merchant_identifier,
            deal_number,
            gross_sales,
            ROW_NUMBER() OVER (PARTITION BY product_merchant_identifier ORDER BY gross_sales DESC) AS rank
        FROM 
            sales_data
    ) AS ranked_deals
    WHERE rank = 1
)
UPDATE da_insights AS rr
SET deal_number = msd.deal_number
FROM max_sales_deal AS msd
WHERE rr.product_merchant_identifier = msd.product_merchant_identifier;

SELECT 
    COUNT(CASE WHEN deal_number IS NULL THEN 1 END) AS null_deal_number
FROM da_insights;

--- bring PC_target

ALTER TABLE da_insights ADD COLUMN pc_target INTEGER;

WITH max_sales_pc AS (
    SELECT 
        product_merchant_identifier,
        pc_target
    FROM (
        SELECT 
            product_merchant_identifier,
            pc_target,
            gross_sales,
            ROW_NUMBER() OVER (PARTITION BY product_merchant_identifier ORDER BY gross_sales DESC) AS rank
        FROM 
            sales_data
    ) AS ranked_deals
    WHERE rank = 1
)
UPDATE da_insights AS rr
SET pc_target = msd.pc_target
FROM max_sales_pc AS msd
WHERE rr.product_merchant_identifier = msd.product_merchant_identifier;

SELECT 
    COUNT(CASE WHEN pc_target IS NULL THEN 1 END) AS null_pc_target
FROM da_insights;

SELECT product_merchant_identifier, COUNT(*)
FROM da_insights
GROUP BY product_merchant_identifier
HAVING COUNT(*) > 1; --- repeted

SELECT *
FROM da_insights
WHERE product_merchant_identifier = '3ZBXWEA9';

DELETE FROM da_insights
WHERE ctid IN (
    SELECT ctid
    FROM (
        SELECT ctid,
               ROW_NUMBER() OVER (PARTITION BY product_merchant_identifier ORDER BY performance_score_ayo DESC) AS fila_numero
        FROM da_insights
    ) AS filas
    WHERE fila_numero > 1
);

SELECT COUNT(*) AS total_registros
FROM da_insights; --- 9027

ALTER TABLE da_insights
ADD COLUMN sale_discount_dec NUMERIC;

UPDATE da_insights
SET sale_discount_dec = sale_discount/100.0;

ALTER TABLE da_insights
ADD COLUMN diff_discounts NUMERIC;

UPDATE da_insights
SET diff_discounts = sale_discount_dec - final_reduction;

ALTER TABLE da_insights
ALTER COLUMN performance_score_ayo TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN price_before_reductions TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN sale_price TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN total_costs TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN return_rate TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN service_fees TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN vat_percentage TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN final_reduction TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN sale_discount_dec TYPE NUMERIC(10, 2);

ALTER TABLE da_insights
ALTER COLUMN diff_discounts TYPE NUMERIC(10, 2);


SELECT final_reduction, COUNT(*) AS frecuencia
FROM da_insights
GROUP BY final_reduction
ORDER BY frecuencia DESC;






