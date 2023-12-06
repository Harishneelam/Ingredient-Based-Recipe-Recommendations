import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components


col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("Cooking-Recipe.png",width = 200)

with col3:
    st.write(' ')



st.title("Ingredient-Based Recipe Recommendations")
df = pd.read_pickle('Recipes_data_Whole_Sim_Ingrs.pkl')
st.write(df)
df_exploded = df.explode('Ingredients')
df_exploded['Value'] = 1
df_pivot = df_exploded.groupby(['Title', 'Ingredients'])['Value'].sum().unstack(fill_value=0).reset_index()
df_pivot.reset_index(inplace=True)
df_boolean = df_pivot.astype(bool)
df_encoded = pd.get_dummies(df_boolean.drop(['Title','index'], axis=1))
columns_to_exclude = ['salt', 'sugar','water', 'vegetable oil']
filtered_columns = [col for col in df_encoded.columns if col not in columns_to_exclude]
df_encoded = df_encoded[filtered_columns]

ing = df_pivot.columns[3:]

frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.2)
def recommend_recipes(ingredients, rules):
    relevant_rules = rules[rules['antecedents'].apply(lambda x: set(ingredients).issubset(set(x)))]
    relevant_rules = relevant_rules.sort_values(by='confidence', ascending=False)

    top_rules = relevant_rules.iloc[0:3]
    recommended_ingredients = list(top_rules['consequents'])

    unique_ingredients = set()
    for item in recommended_ingredients:
        unique_ingredients.update(item)
    
    return list(unique_ingredients)

ing_input = st.multiselect("Select your preferred Ingredients", ing,default=['garlic','tomatoes','butter'])
user_input_ingredients = list(ing_input)

recommended_ingredients = []
for ing in user_input_ingredients:
    rins = recommend_recipes([ing], rules)
    for i in rins:
        if(i not in recommended_ingredients and i not in user_input_ingredients):
            recommended_ingredients.append(i)

combs = []
for j in range(1, len(recommended_ingredients) + 1):
    combs.extend(list(combinations(recommended_ingredients, j)))

recoms = [user_input_ingredients]
for i in combs:
    recoms.append(user_input_ingredients + list(i))



recommended_recipes = pd.DataFrame()
for rc in recoms:
    recommended_recipes = pd.concat([recommended_recipes, df[df['Ingredients'].apply(lambda x: all(ingredient in x for ingredient in rc))]])


recommended_recipes= recommended_recipes.reset_index(drop=True)



Ings = user_input_ingredients + recommended_ingredients
import pickle

with open('sgd_classifier_Cuisines.pkl', 'rb') as model_file:
     sgd_classifier_Cuisines = pickle.load(model_file)

with open('sgd_classifier_Course.pkl', 'rb') as model_file:
     sgd_classifier_Course = pickle.load(model_file)

with open('sgd_classifier_Dietary.pkl', 'rb') as model_file:
     sgd_classifier_Dietary = pickle.load(model_file)

Ings_str = ' '.join(Ings)
classes =[]

from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_pickle('Recipes_data_Cuisines_Sim_Ingrs.pkl')
df['Ingredients'] = df['Ingredients'].apply(lambda x: ' '.join(x))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Ingredients'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
Ings_str = ' '.join(Ings)

tfidf_features = tfidf_vectorizer.transform([Ings_str])

class_probabilities = sgd_classifier_Cuisines.predict_proba(tfidf_features)

top_indices = [i for i, prob in enumerate(class_probabilities[0]) if prob > 0.25]

top_classes = sgd_classifier_Cuisines.classes_[top_indices]

if len(top_classes)==0:
    top_indices = class_probabilities.argsort()[0][-1:][::-1]
    top_classes = sgd_classifier_Cuisines.classes_[top_indices]

classes.extend(top_classes)

df = pd.read_pickle('Recipes_data_Dietary_Sim_Ingrs.pkl')
df['Ingredients'] = df['Ingredients'].apply(lambda x: ' '.join(x))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Ingredients'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
Ings_str = ' '.join(Ings)

tfidf_features = tfidf_vectorizer.transform([Ings_str])

class_probabilities = sgd_classifier_Dietary.predict_proba(tfidf_features)

top_indices = [i for i, prob in enumerate(class_probabilities[0]) if prob > 0.25]

# Get the corresponding class labels
top_classes = sgd_classifier_Dietary.classes_[top_indices]

if len(top_classes)==0:
    top_indices = class_probabilities.argsort()[0][-1:][::-1]
    top_classes = sgd_classifier_Dietary.classes_[top_indices]

classes.extend(top_classes)

df = pd.read_pickle('Recipes_data_Course_Sim_Ingrs.pkl')
df['Ingredients'] = df['Ingredients'].apply(lambda x: ' '.join(x))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Ingredients'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
Ings_str = ' '.join(Ings)

tfidf_features = tfidf_vectorizer.transform([Ings_str])

class_probabilities = sgd_classifier_Course.predict_proba(tfidf_features)

top_indices = [i for i, prob in enumerate(class_probabilities[0]) if prob > 0.25]


top_classes = sgd_classifier_Course.classes_[top_indices]

if len(top_classes)==0:
    top_indices = class_probabilities.argsort()[0][-1:][::-1]
    top_classes = sgd_classifier_Course.classes_[top_indices]

classes.extend(top_classes)

recommended_recipes = recommended_recipes[recommended_recipes['Type'].isin(classes)].reset_index(drop=True)

st.write(f"Here are your {len(recommended_recipes)} Recommended recipes:")
st.dataframe(recommended_recipes)



vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
X = vectorizer.fit_transform(recommended_recipes['Ingredients'])

co_occurrence_matrix = (X.T * X)
co_occurrence_df = pd.DataFrame(co_occurrence_matrix.A, index=vectorizer.get_feature_names_out(), columns=vectorizer.get_feature_names_out())
co_occurrence_array = co_occurrence_df.values

np.fill_diagonal(co_occurrence_array, 0)
co_occurrence_df = pd.DataFrame(co_occurrence_array, index=co_occurrence_df.index, columns=co_occurrence_df.columns)


all_zero_rows = (co_occurrence_df == 0).all(axis=1)
co_occurrence_df_filtered = co_occurrence_df[~all_zero_rows]

all_zero_columns = (co_occurrence_df_filtered == 0).all(axis=0)
co_occurrence_dfd = co_occurrence_df_filtered.loc[:, ~all_zero_columns]

def color_legend(color, label):
        return f'<div style="display: flex; align-items: center; margin-right: 10px;"><div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; margin-right: 5px;"></div><span>{label}</span></div>'

def build_network(recommended_recipes):
    clusters = np.unique(recommended_recipes['Type'])
    my_dict = {k: [] for k in clusters}

    for i in range(len(recommended_recipes['Ingredients'])):
        cuisine = recommended_recipes['Type'].iloc[i]
        ingredients = recommended_recipes['Ingredients'].iloc[i]
        
        my_dict[cuisine].extend(ingredients)

    my_dict = {k: list(v) for k, v in my_dict.items()}

    

    Rd_colors = [
    "#ADD8E6",  # Light Blue
    "#90EE90",  # Light Green
    "#FFFFE0",  # Light Yellow
    "#E6E6FA",  # Light Purple
    "#FF6347",  # Tomato
    "#8A2BE2",  # Blue Violet
    "#4B0082",  # Indigo
    "#800000",  # Maroon
    "#483D8B",  # Dark Slate Blue
    "#FF0000",  # Red
    "#FF1493",  # Deep Pink
    "#B22222",  # Fire Brick
    "#008000",  # Green
    "#0000CD",  # Medium Blue
    "#0000FF"   # Blue
    "#FFD700",  # Gold
    "#DAA520",  # Goldenrod
    "#9932CC",  # Dark Orchid
    "#8B4513",  # Saddle Brown
    "#48D1CC",  # Medium Turquoise
    "#DC143C",  # Crimson
    "#556B2F",  # Dark Olive Green
    "#008080"   # Teal
    ]

    cls = np.random.choice(len(Rd_colors), len(my_dict), replace=False)
    color_mapping = {cuisine: Rd_colors[index] for cuisine, index in zip(my_dict.keys(), cls)}

    G = nx.Graph()
    for col in co_occurrence_dfd.columns:
        clr = 'gray'
        count = 0
        for key in my_dict.keys():
            if col in my_dict[key]:
                if(my_dict[key].count(col) > count):
                    count = my_dict[key].count(col)
                    clr = color_mapping.get(key, 'gray')
        G.add_node(col, color = clr)
        for index, value in co_occurrence_dfd[col].items():
            if value > 0 and col != index:
                G.add_edge(col, index)

    
    degree_centrality = nx.degree_centrality(G)

    net = Network(height="750px", width="100%", bgcolor='black', font_color='white',notebook=True)


    for node in G.nodes:
        color = G.nodes[node]['color']
        size = 10 + recommended_recipes.shape[0] * degree_centrality[node]  
        net.add_node(node, color=color, size=size) 

 
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])



    net.options.physics.enabled = True
    net.options.physics.barnesHut.gravitationalConstant = -145000
    net.options.physics.barnesHut.centralGravity = 2.9
    net.options.physics.barnesHut.springLength = 400
    net.options.physics.barnesHut.springConstant = 0.03
    net.options.physics.barnesHut.damping = 0.85

    net.options.physics.timestep = 0.1  

    net.show_buttons(filter_=['manipulation'])
    net.show("network.html")

    return color_mapping



color_mapping = build_network(recommended_recipes)

st.write("Click the below button to see how your ingredient network looks based on your recommended recipes!!")

button_clicked = st.checkbox("Show Network")

if button_clicked:
   
    with open("network.html", "r") as html_file:
        html_content = html_file.read()
        components.html(html_content, width=800, height=1000)

legend_html = ''.join([color_legend(color, label) for color, label in zip(color_mapping.values(), color_mapping.keys())])
st.markdown(legend_html, unsafe_allow_html=True)

new_ing =  list(recommended_recipes.explode('Ingredients').Ingredients)


st.write("Do you want to customize your recipes further?")
yes = st.checkbox('Yes',value=False)
filtered_recipes = recommended_recipes.copy()
if yes:
    no_list = list(st.multiselect("Enter the Ingredients you don't want in your recipes.", new_ing))
    filtered_recipes = filtered_recipes[~filtered_recipes['Ingredients'].apply(lambda x: any(ingredient in x for ingredient in no_list))].reset_index(drop=True)

st.write("Do you want a specific category of your recommended recipes?")
yeah = st.checkbox("Yeah")

if yeah:
    st.write("Please select the category/categories")
    unique_types = np.unique(filtered_recipes['Type'])
    selected_types = [st.checkbox(type, value=False) for type in unique_types]
    filtered_recipes = filtered_recipes[filtered_recipes['Type'].isin([unique_types[i] for i, selected in enumerate(selected_types) if selected])].reset_index(drop=True)

if yes or yeah:  
    filtered_recipes = filtered_recipes.reset_index(drop=True)
    st.write(f"Here are your customized {len(filtered_recipes)} recipes:")    
    st.dataframe(filtered_recipes)
    type_counts = pd.DataFrame(filtered_recipes['Type'].value_counts())

   
    fig = px.pie(type_counts, names=type_counts.index, values='count', title='Distribution of Recipe Types',
                color_discrete_sequence=px.colors.sequential.RdBu, hole=0.3)

    
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])  
    fig.update_layout(showlegend=False, width=800, height=600)


    st.plotly_chart(fig)
