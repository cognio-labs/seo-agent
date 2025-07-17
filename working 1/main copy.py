# main.py - Streamlit Frontend for SEO Agent System

import streamlit as st
import json
import pandas as pd
from datetime import datetime
import time
from agent import run_full_workflow
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="SEO Content Agent System",
                   page_icon="📝",
                   layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .node-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .node-active {
        background-color: #FFF3CD;
        border: 1px solid #FFC107;
    }
    .node-complete {
        background-color: #D4EDDA;
        border: 1px solid #28A745;
    }
    .node-pending {
        background-color: #F8F9FA;
        border: 1px solid #DEE2E6;
    }
</style>
""",
            unsafe_allow_html=True)

# Initialize session state
if 'workflow_running' not in st.session_state:
    st.session_state.workflow_running = False
if 'workflow_results' not in st.session_state:
    st.session_state.workflow_results = None
if 'current_node' not in st.session_state:
    st.session_state.current_node = None
if 'node_outputs' not in st.session_state:
    st.session_state.node_outputs = {}

# Header
st.title("🚀 SEO Content Agent System")
st.markdown("### Generate SEO-optimized content with AI-powered workflow")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Check API keys
    st.subheader("API Keys Status")
    api_keys = {
        "GROQ_API_KEY":
        "✅" if os.getenv("GROQ_API_KEY") else "❌",
        "OPENAI_API_KEY":
        "✅" if os.getenv("OPENAI_API_KEY") else "❌",
        "GOOGLE_API_KEY":
        "✅" if os.getenv("GOOGLE_API_KEY") else "❌",
        "SERPAPI_API_KEY":
        "✅" if os.getenv("SERPAPI_API_KEY") else "❌",
        "KEYWORD_EVERYWHERE_API_KEY":
        "✅" if os.getenv("KEYWORD_EVERYWHERE_API_KEY") else "❌",
        "YOUTUBE_API_KEY":
        "✅" if os.getenv("YOUTUBE_API_KEY") else "❌",
        "REPLICATE_API_TOKEN":
        "✅" if os.getenv("REPLICATE_API_TOKEN") else "❌"
    }

    for key, status in api_keys.items():
        st.write(f"{status} {key}")

    st.divider()

    # User preferences
    st.subheader("Content Strategy")

    frequency = st.selectbox("Publishing Frequency",
                             ["daily", "weekly", "monthly"],
                             index=1)

    business_focus = st.selectbox("Business Focus",
                                  ["traffic", "authority", "product"],
                                  index=0)

    time_horizon = st.selectbox("Time Horizon", [30, 60, 90],
                                format_func=lambda x: f"{x} days",
                                index=0)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Workflow Input")

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Enter keywords manually", "Upload CSV file", "Upload Excel file"])

    initial_query = ""

    if input_method == "Enter keywords manually":
        initial_query = st.text_area(
            "Enter keywords (comma-separated):",
            placeholder=
            "SEO tools, content marketing, keyword research, link building",
            height=100)
    elif input_method == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open("temp_keywords.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            initial_query = "temp_keywords.csv"
            st.success("CSV file uploaded successfully!")
    else:  # Excel file
        uploaded_file = st.file_uploader("Choose an Excel file",
                                         type=["xlsx", "xls"])
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open("temp_keywords.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            initial_query = "temp_keywords.xlsx"
            st.success("Excel file uploaded successfully!")

    # Start workflow button
    if st.button("🚀 Start SEO Workflow",
                 type="primary",
                 disabled=st.session_state.workflow_running
                 or not initial_query):
        st.session_state.workflow_running = True
        st.session_state.node_outputs = {}
        st.session_state.current_node = None

with col2:
    st.header("🔄 Workflow Status")

    # Define workflow nodes for status display
    workflow_nodes = [("get_initial_keywords", "📥 Get Initial Keywords"),
                      ("enrich_keywords", "💎 Enrich Keywords"),
                      ("cluster_keywords", "🎯 Cluster Keywords"),
                      ("request_user_preferences", "⚙️ Set Preferences"),
                      ("score_and_tag_topics", "📊 Score Topics"),
                      ("generate_editorial_calendar", "📅 Generate Calendar"),
                      ("generate_content_briefs", "📝 Generate Briefs"),
                      ("serp_analysis", "🔍 SERP Analysis"),
                      ("entity_semantic_optimization",
                       "🧠 Semantic Optimization"),
                      ("create_blueprint", "📐 Create Blueprint"),
                      ("write_article", "✍️ Write Article"),
                      ("verify_article", "✅ Verify Article"),
                      ("media_planner", "🎨 Plan Media"),
                      ("compile_final", "📦 Compile Final Article")]

    # Display node status
    for node_id, node_name in workflow_nodes:
        if node_id in st.session_state.node_outputs:
            st.markdown(
                f'<div class="node-status node-complete">{node_name} ✓</div>',
                unsafe_allow_html=True)
        elif st.session_state.current_node == node_id:
            st.markdown(
                f'<div class="node-status node-active">{node_name} ⏳</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="node-status node-pending">{node_name}</div>',
                unsafe_allow_html=True)

# Results area
st.divider()
results_container = st.container()

# Run workflow when button is clicked
if st.session_state.workflow_running and initial_query:
    with results_container:
        st.header("📈 Workflow Progress")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create tabs for different outputs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Keywords & Clusters", "📊 Topic Scoring", "📅 Editorial Calendar",
            "📝 Content Briefs", "📄 Final Articles"
        ])

        # Run the workflow
        user_preferences = {
            "frequency": frequency,
            "business_focus": business_focus,
            "time_horizon": time_horizon
        }

        try:
            node_count = 0
            total_nodes = len(workflow_nodes)

            for state in run_full_workflow(initial_query, user_preferences):
                # Update current node
                for key in state.keys():
                    if key not in st.session_state.node_outputs:
                        st.session_state.current_node = key
                        st.session_state.node_outputs[key] = state.get(key)
                        node_count = len(st.session_state.node_outputs)
                        break

                # Update progress
                progress = node_count / total_nodes
                progress_bar.progress(progress)

                # Display current status
                current_node_name = next(
                    (name for nid, name in workflow_nodes
                     if nid == st.session_state.current_node), "Processing...")
                status_text.text(f"Current Step: {current_node_name}")

                # Update tabs with results
                with tab1:
                    if 'keywords' in state and state['keywords']:
                        st.subheader("📌 Initial Keywords")
                        st.write(f"Found {len(state['keywords'])} keywords")
                        with st.expander("View Keywords"):
                            st.write(state['keywords'])

                    if 'enriched_keywords' in state and state[
                            'enriched_keywords']:
                        st.subheader("💎 Enriched Keywords")
                        df = pd.DataFrame(state['enriched_keywords'])
                        st.dataframe(df, use_container_width=True)

                    if 'keyword_clusters' in state and state[
                            'keyword_clusters']:
                        st.subheader("🎯 Keyword Clusters")
                        for cluster in state['keyword_clusters']:
                            with st.expander(
                                    f"Cluster: {cluster['topic_title']}"):
                                st.write("Keywords:",
                                         ", ".join(cluster['keywords']))

                with tab2:
                    if 'scored_and_tagged_topics' in state and state[
                            'scored_and_tagged_topics']:
                        st.subheader("📊 Scored Topics")
                        topics_df = pd.DataFrame([{
                            "Topic":
                            topic['topic_title'],
                            "COS Score":
                            topic['cos'],
                            "Tags":
                            topic['priority_tags'],
                            "Keywords":
                            len(topic['keywords'])
                        } for topic in state['scored_and_tagged_topics']])
                        st.dataframe(topics_df, use_container_width=True)

                with tab3:
                    if 'content_plan' in state and state['content_plan']:
                        st.subheader("📅 Editorial Calendar")
                        calendar_data = []
                        current_date = datetime.now()
                        for i, topic in enumerate(state['content_plan']):
                            calendar_data.append({
                                "Publish Date": (current_date.replace(
                                    day=1) if i == 0 else current_date.replace(
                                        day=i + 1)).strftime("%Y-%m-%d"),
                                "Topic":
                                topic['topic_title'],
                                "Score":
                                topic['cos'],
                                "Tags":
                                topic['priority_tags']
                            })
                        calendar_df = pd.DataFrame(calendar_data)
                        st.dataframe(calendar_df, use_container_width=True)

                with tab4:
                    if 'content_briefs' in state and state['content_briefs']:
                        st.subheader("📝 Content Briefs")
                        for i, brief in enumerate(state['content_briefs']):
                            with st.expander(
                                    f"Brief {i+1}: {brief.get('topic', 'N/A')}"
                            ):
                                st.json(brief)

                with tab5:
                    if 'written_article' in state and state['written_article']:
                        st.subheader("✍️ Current Article Being Written")
                        with st.expander("View Article Draft"):
                            st.markdown(state['written_article'][:1000] +
                                        "..." if len(state['written_article'])
                                        > 1000 else state['written_article'])

                    if 'verification_results' in state and state[
                            'verification_results']:
                        st.subheader("✅ Verification Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            score = state['verification_results'].get(
                                'score', 0)
                            st.metric("SEO Score", f"{score}/100")
                        with col2:
                            approved = state['verification_results'].get(
                                'approved', False)
                            st.metric(
                                "Status", "Approved ✅"
                                if approved else "Needs Revision ⚠️")

                        if state['verification_results'].get('issues'):
                            st.warning("Issues Found:")
                            for issue in state['verification_results'][
                                    'issues']:
                                st.write(f"- {issue}")

                    if 'media_plan' in state and state['media_plan']:
                        st.subheader("🎨 Media Plan")
                        if state['media_plan'].get('youtube_videos'):
                            st.write("**YouTube Videos:**")
                            for video in state['media_plan']['youtube_videos']:
                                st.write(f"- {video['title']}")

                        if state['media_plan'].get('generated_images'):
                            st.write("**Generated Images:**")
                            for img in state['media_plan']['generated_images']:
                                st.write(f"- {img['prompt'][:50]}...")

                    if 'completed_articles' in state and state[
                            'completed_articles']:
                        st.subheader("📄 Completed Articles")
                        for i, article_data in enumerate(
                                state['completed_articles']):
                            brief = article_data['brief']
                            article = article_data['article']
                            score = article_data['verification_score']

                            with st.expander(
                                    f"Article {i+1}: {brief.get('topic', 'N/A')} (Score: {score}/100)"
                            ):
                                st.markdown(article)

                                # Download button
                                st.download_button(
                                    label="📥 Download Article",
                                    data=article,
                                    file_name=
                                    f"{brief.get('suggested_slug', f'article_{i+1}')}.md",
                                    mime="text/markdown",
                                    key=f"download_{i}")

                # Small delay to show progress
                time.sleep(0.1)

            # Workflow completed
            progress_bar.progress(1.0)
            status_text.text("✅ Workflow Completed!")
            st.success("🎉 All articles have been generated successfully!")

            # Save final results
            st.session_state.workflow_results = state
            st.session_state.workflow_running = False

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.session_state.workflow_running = False

# Display saved results if available
elif st.session_state.workflow_results and not st.session_state.workflow_running:
    with results_container:
        st.header("📊 Previous Results")
        st.info("Results from the last workflow run are displayed below.")

        if 'completed_articles' in st.session_state.workflow_results:
            st.subheader("📄 Generated Articles")
            for i, article_data in enumerate(
                    st.session_state.workflow_results['completed_articles']):
                brief = article_data['brief']
                article = article_data['article']
                score = article_data['verification_score']

                with st.expander(
                        f"Article {i+1}: {brief.get('topic', 'N/A')} (Score: {score}/100)"
                ):
                    st.markdown(article)

                    # Download button
                    st.download_button(
                        label="📥 Download Article",
                        data=article,
                        file_name=
                        f"{brief.get('suggested_slug', f'article_{i+1}')}.md",
                        mime="text/markdown",
                        key=f"download_saved_{i}")
