import streamlit as st
import pandas as pd
import numpy as np
from client import name_topic

st.write("Hello World")
# st.write("## This is a H2 Title!")
# x = st.text_input("Movie", "Star Wars")

# if st.button("Click Me"):
#     st.write(f"Your favorite movie is `{x}`")


# data = pd.read_csv("movies.csv")
# st.write(data)


# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

# st.bar_chart(chart_data)



# # Setting the API key

# # Topic #1
# st.write(name_topic('right eat big look car play going let come little cat baby good dog goes went ate milk school red'))

# # Topic #2
# st.write(name_topic('look right come let going little said good ate say wait hungry thing big play hear eat way watch think'))

# # Topic #3
# st.write(name_topic('baby need right eat juice let sit help look pizza bubbles open bubble play apple big food going blue milk'))
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.decomposition import NMF

# # Dummy data for demonstration (replace with your actual data)
# documents = {
#     21: "apple juice toy play",
#     24: "milk cookies run fun",
#     27: "toy car book story",
#     30: "school work teacher friend",
#     33: "story fun play reading",
#     36: "run fast jump high"
# }

# # Sidebar for page navigation
# st.sidebar.title("Navigation")
# pages = ["Home", "Age Bucket Analysis", "Topic Analysis", "Configurable Topic Modeling"]
# selected_page = st.sidebar.radio("Go to", pages)

# if selected_page == "Configurable Topic Modeling":
#     st.title("Configurable Topic Modeling")
#     st.markdown("""
#     Customize the topic modeling parameters and explore the results dynamically.
#     """)

#     # User Inputs
#     n_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=3, step=1)
#     n_top_words = st.slider("Number of Top Words per Topic", min_value=5, max_value=20, value=10, step=1)
#     algorithm = st.selectbox("Choose Topic Modeling Algorithm", ["LatentDirichletAllocation", "NMF"])
    
#     # Extract ages and document texts
#     ages = sorted(documents.keys())
#     document_texts = [documents[age] for age in ages]
    
#     # Step 1: Vectorize the text data
#     vectorizer = CountVectorizer(stop_words="english")
#     doc_term_matrix = vectorizer.fit_transform(document_texts)
#     feature_names = vectorizer.get_feature_names_out()

#     # Step 2: Apply chosen algorithm
#     if algorithm == "LatentDirichletAllocation":
#         model = LatentDirichletAllocation(n_components=n_topics, random_state=0)
#     elif algorithm == "NMF":
#         model = NMF(n_components=n_topics, random_state=0)
    
#     model.fit(doc_term_matrix)
        
#     # col1, col2 = st.columns([1, 1])
#     st.subheader("Discovered Topics")

#     on = st.toggle("Activate feature")

#     # Display Topics
#     # st.subheader("Discovered Topics")
#     # on = st.toggle("Activate feature")
#     topics = {}
#     for topic_idx, topic in enumerate(model.components_):
#         top_features_indices = topic.argsort()[-n_top_words:][::-1]
#         top_features = [feature_names[i] for i in top_features_indices]
#         topics[f"Topic {topic_idx + 1}"] = ", ".join(top_features)
#         st.write(f"**Topic {topic_idx + 1}**: {', '.join(top_features)}")
    
#     # Step 3: Get topic distribution for each document
#     doc_topic_distributions = model.transform(doc_term_matrix)
#     df_topic_proportions = pd.DataFrame(doc_topic_distributions, index=ages, columns=[f"Topic {i+1}" for i in range(n_topics)])
    
#     # Step 4: Apply a moving average for smoothing
#     window_size = st.slider("Smoothing Window Size", min_value=1, max_value=5, value=3, step=1)
#     df_topic_proportions_smoothed = df_topic_proportions.rolling(window=window_size, center=True).mean()

#     # Visualization
#     st.subheader("Smoothed Topic Proportions Across Ages")
#     st.line_chart(df_topic_proportions_smoothed)
