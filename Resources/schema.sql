--
-- PostgreSQL database dump
--

-- Dumped from database version 16.4
-- Dumped by pg_dump version 16.4

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: valorant_agent_stats; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.valorant_agent_stats (
    id integer NOT NULL,
    map character varying(50),
    rank character varying(50),
    agent character varying(50),
    role character varying(50),
    winrate double precision,
    totalmatches integer
);


ALTER TABLE public.valorant_agent_stats OWNER TO postgres;

--
-- Name: valorant_agent_stats_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.valorant_agent_stats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.valorant_agent_stats_id_seq OWNER TO postgres;

--
-- Name: valorant_agent_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.valorant_agent_stats_id_seq OWNED BY public.valorant_agent_stats.id;


--
-- Name: valorant_agent_stats id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.valorant_agent_stats ALTER COLUMN id SET DEFAULT nextval('public.valorant_agent_stats_id_seq'::regclass);


--
-- Name: valorant_agent_stats valorant_agent_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.valorant_agent_stats
    ADD CONSTRAINT valorant_agent_stats_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

