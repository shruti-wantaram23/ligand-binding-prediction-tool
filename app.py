import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.title("Ligand Binding Prediction Tool")

Energy = st.number_input("Enter Binding_energy")
hydrogen_bonds = st.number_input("Enter Hydrogen_bonds")

model = joblib.load("model.pkl")


def classify_energy(energy):
    if energy => -9.5:
        return "Strong"
    elif energy => -8:
        return "Moderate"
    
if st.button("Predict"):

    result = model.predict([[Energy, hydrogen_bonds]])
    rule_based = classify_energy(Energy)

   
    if rule_based == "Strong":
        st.success(f"Prediction: {rule_based}")
    elif rule_based == "Moderate":
        st.warning(f"Prediction: {rule_based}")
    else:
        st.error(f"Prediction: {rule_based}")

   
    st.write("Model vs Rule")
    st.write("ML Prediction:", result[0])
    st.write("Rule-based:", rule_based)

    
    st.write("Input Summary")
    st.write("Binding energy:", Energy)
    st.write("Hydrogen Bonds:", hydrogen_bonds)

st.header("Data Analysis")

data = pd.read_csv("Dock_results.csv")

data["Class"] = data["Binding_energy"].apply(classify_energy)

st.write("Binding Energy Graph")

colors = data["Class"].map({
    "Strong": "green",
    "Moderate": "orange"})

fig1, ax1 = plt.subplots()
ax1.bar(data["Ligand"], data["Binding_energy"], color=colors)
ax1.set_xlabel("Ligand")
ax1.set_ylabel("Binding Energy")
ax1.set_title("Binding Energy by Ligand")

st.pyplot(fig1)


st.write("Boxplot")

fig2, ax2 = plt.subplots()
ax2.boxplot(data["Binding_energy"])
ax2.set_title("Binding Energy Spread")

st.pyplot(fig2)

st.write("PCA Analysis")


features = data[["Binding_energy", "Hydrogen_bonds"]]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Class"] = data["Class"]

fig3, ax3 = plt.subplots()

for cls in pca_df["Class"].unique():
    subset = pca_df[pca_df["Class"] == cls]
    ax3.scatter(subset["PC1"], subset["PC2"], label=cls)

ax3.set_xlabel("Principal Component 1")
ax3.set_ylabel("Principal Component 2")
ax3.set_title("PCA of Ligands")
ax3.legend()

st.pyplot(fig3)

# Create buffer
import io
buf = io.BytesIO()

# Save figure into buffer
fig1.savefig(buf, format="png")
buf.seek(0)

# Download button
st.download_button(
    label="⬇️ Download Graph",
    data=buf,
    file_name="binding_energy_graph.png",
    mime="image/png"
)

buf2 = io.BytesIO()
fig2.savefig(buf2, format="png")
buf2.seek(0)

st.download_button(
    label="⬇️ Download Boxplot",
    data=buf2,
    file_name="boxplot.png",
    mime="image/png"
)

import streamlit as st

# ---------------- PAGE NAVIGATION ----------------
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Prediction", "Analysis"]
)

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("Ligand Binding Prediction Tool")

    st.write("""This tool predicts drug-binding affinity for viral proteins using machine learning.""")

    st.header("Applications")
    st.write  ("""

    • Drug Discovery & Virtual Screening
      This tool helps in rapid screening of thousands of compounds against viral proteins like Nipah virus glycoprotein. 

    • Antiviral Research (Nipah Virus Focus)
     The tool is highly useful for emerging viral diseases like Nipah virus.
     Identifies compounds that bind strongly to viral proteins 
 
    • Decision Support System for Researchers
      Acts as a decision-making tool before wet-lab experiments.
      Ranks compounds based on predicted binding strength
      Helps researchers prioritize best candidates
      Saves:Time and Cost

   • Data Analysis & Pattern Discovery (Using PCA)
     PCA (Principal Component Analysis) helps in understanding hidden patterns in docking data.
     Reduces multiple features → 2D visualization
     Shows clustering of:
     Strong binders
     Weak binders
     Helps identify:
     Feature importance
     Correlation between energy & bonds""")


# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
    st.title("Prediction Page")
    st.write("Model prediction will go here")

# ---------------- ANALYSIS PAGE ----------------
elif page == "Analysis":
    st.title("Data Analysis")
    st.write("Graphs will go here")
