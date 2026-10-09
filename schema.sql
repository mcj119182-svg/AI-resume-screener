-- ============================================================
-- AI 智能简历筛选器 —— 数据库初始化脚本
-- 数据库：MySQL 8.0+
-- 用法：mysql -u root -p < schema.sql
--
-- ⚠️ 时间列名是 create_time（不是 created_at）。
--    Agent 的查重逻辑要读这一列来判断「最近 N 天内是否评估过」，
--    列名不一致会导致查重失效，且不会报错（会静默降级）。
-- ============================================================

CREATE DATABASE IF NOT EXISTS recruit_db
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE recruit_db;

-- 候选人打分结果表：列名与 ai_screener.save_to_database 的 INSERT 语句一一对应
CREATE TABLE IF NOT EXISTS candidates (
    id               BIGINT UNSIGNED   NOT NULL AUTO_INCREMENT COMMENT '主键',
    name             VARCHAR(50)       NOT NULL                COMMENT '候选人姓名（服务端已限制 <=40 字符）',
    skills           VARCHAR(200)               DEFAULT NULL  COMMENT '核心技能，逗号分隔（服务端已限制 <=500 字符）',
    match_score      INT               NOT NULL                COMMENT '匹配度 0-100（服务端已钳制范围）',
    ai_review        VARCHAR(500)               DEFAULT NULL  COMMENT 'AI 一句话点评（服务端已限制 <=500 字符）',
    resume_file_path VARCHAR(200)               DEFAULT NULL  COMMENT '来源标记：本地文件路径 / API接口调用 / Streamlit网页调用',
    resume_hash      CHAR(64)                   DEFAULT NULL  COMMENT '简历指纹：规范化（去空白/换行/标点）后内容的 SHA-256，用于查重',
    create_time      DATETIME                   DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间，查重与限流依赖这一列',
    PRIMARY KEY (id),
    KEY idx_score (match_score),
    KEY idx_name_time (name, create_time),
    KEY idx_name_hash (name, resume_hash)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = 'AI 简历筛选结果';

-- 被拒绝 / 被跳过的请求表：用于统计误拒率
--   reason 前缀约定：
--     ai_rejected:*  AI 判定不像简历 / 分数为 0 / 注入 / 格式异常
--     skipped:*      命中查重规则，自动跳过
CREATE TABLE IF NOT EXISTS rejected_logs (
    id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    name        VARCHAR(50)              DEFAULT NULL COMMENT '候选人姓名',
    reason      VARCHAR(255)    NOT NULL             COMMENT '拒绝原因，如 ai_rejected:zero_score / skipped:recent_high_score',
    resume_hash CHAR(64)                 DEFAULT NULL COMMENT '简历指纹（规范化后 SHA-256）',
    raw_score   INT                      DEFAULT NULL COMMENT 'AI 原始分数（没有则为 NULL）',
    create_time DATETIME                 DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
    PRIMARY KEY (id),
    KEY idx_name_time (name, create_time),
    KEY idx_hash (resume_hash)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '被拒绝/被跳过的请求，用于统计误拒率';

-- 索引说明：
--   idx_score      给历史记录页的 ORDER BY match_score DESC LIMIT 500 用
--   idx_name_time  给查重用的 WHERE name = ? ORDER BY create_time DESC 用
--                  （同时服务同名限流：WHERE name = ? AND create_time >= ?）
--   idx_name_hash  给指纹比对的 WHERE name = ? AND resume_hash = ? 用

-- ------------------------------------------------------------
-- 如果表已经建过（旧版本），需要增量升级：
--   ALTER TABLE candidates ADD COLUMN resume_hash CHAR(64) DEFAULT NULL;
--   CREATE TABLE IF NOT EXISTS rejected_logs (...);
-- 推荐直接运行 python ai_screener.py --migrate 自动补（也可重启一次服务自愈）。
-- ------------------------------------------------------------
-- 只缺时间列时可单独补：
--   ALTER TABLE candidates
--       ADD COLUMN create_time DATETIME DEFAULT CURRENT_TIMESTAMP;
--   CREATE INDEX idx_name_time ON candidates (name, create_time);
-- ------------------------------------------------------------
