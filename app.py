import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

# ---------------- FUNCTION ----------------

def classify_energy(energy):
    if energy <= -9.5:
        return "Strong"
    elif -9.5 < energy <= -8:
        return "Moderate"
    else:
        return "Weak"

# ---------------- PAGE NAVIGATION ----------------

page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Prediction", "Analysis"]
)

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.title("Ligand Binding Prediction Tool")

    st.write("""
    This tool predicts drug-binding affinity for viral proteins
    using machine learning.
    """)

    st.header("Applications")

    st.write("""

• Drug Discovery & Virtual Screening  
Rapid screening of compounds against viral proteins.

• Antiviral Research  
Useful for emerging viruses like Nipah virus.

• Decision Support System  
Helps prioritize compounds before wet-lab testing.

• Data Analysis Using PCA  
Shows clustering of strong and weak binders.
""")

# ---------------- PREDICTION PAGE ----------------

elif page == "Prediction":

    st.title("Ligand Binding Prediction")

    Energy = st.number_input("Enter Binding Energy")
    hydrogen_bonds = st.number_input("Enter Hydrogen Bonds")

    # Load model here
    model = joblib.load("model.pkl")

    if st.button("Predict"):

        result = model.predict(
            [[Energy, hydrogen_bonds]]
        )

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

        st.write("Binding Energy:", Energy)
        st.write("Hydrogen Bonds:", hydrogen_bonds)

# ---------------- ANALYSIS PAGE ----------------

elif page == "Analysis":

    st.title("Data Analysis")

    data = pd.read_csv("Dock_results.csv")

    data["Class"] = data["Binding_energy"].apply(
        classify_energy
    )

    # -------- BAR GRAPH --------

    st.subheader("Binding Energy Graph")

    colors = data["Class"].map({
        "Strong": "green",
        "Moderate": "orange",
        "Weak": "red"
    })

    fig1, ax1 = plt.subplots()

    ax1.bar(
        data["Ligand"],
        data["Binding_energy"],
        color=colors
    )

    ax1.set_xlabel("Ligand")
    ax1.set_ylabel("Binding Energy")
    ax1.set_title("Binding Energy by Ligand")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    st.pyplot(fig1)

    # -------- BOX PLOT --------

    st.subheader("Boxplot")

    fig2, ax2 = plt.subplots()

    ax2.boxplot(
        data["Binding_energy"]
    )

    ax2.set_title(
        "Binding Energy Spread"
    )

    st.pyplot(fig2)

    # -------- PCA --------

    st.subheader("PCA Analysis")

    features = data[
        ["Binding_energy", "Hydrogen_bonds"]
    ]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        features
    )

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(
        scaled_data
    )

    pca_df = pd.DataFrame(
        pca_result,
        columns=["PC1", "PC2"]
    )

    pca_df["Class"] = data["Class"]

    fig3, ax3 = plt.subplots()

    for cls in pca_df["Class"].unique():

        subset = pca_df[
            pca_df["Class"] == cls
        ]

        ax3.scatter(
            subset["PC1"],
            subset["PC2"],
            label=cls
        )

    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("PCA of Ligands")
    ax3.legend()

    st.pyplot(fig3)

    # -------- DOWNLOAD --------

    buf = io.BytesIO()

    fig1.savefig(buf, format="png")
    buf.seek(0)

    st.download_button(
        label="⬇️ Download Graph",
        data=buf,
        file_name="binding_energy_graph.png",
        mime="image/png"
    )
