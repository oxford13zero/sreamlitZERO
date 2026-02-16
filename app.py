import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

st.set_page_config(page_title="TECH4ZERO Debug", layout="wide")

st.title("🔍 COMPREHENSIVE DEBUG - TECH4ZERO-MX")

# ══════════════════════════════════════════════════════════════
# 1. SYSTEM INFO
# ══════════════════════════════════════════════════════════════
st.header("1️⃣ System Information")
st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")
st.write(f"Pandas version: {pd.__version__}")
st.write(f"NumPy version: {np.__version__}")

# ══════════════════════════════════════════════════════════════
# 2. ENVIRONMENT VARIABLES (os.environ)
# ══════════════════════════════════════════════════════════════
st.header("2️⃣ Environment Variables (os.environ)")

env_keys = [k for k in os.environ.keys() if any(x in k.upper() for x in ['SUPABASE', 'GROQ', 'JWT', 'VERCEL'])]
st.write(f"Found {len(env_keys)} relevant environment variables")

env_dict = {}
for k in sorted(env_keys):
    val = os.environ.get(k, "")
    if val and len(val) > 20:
        env_dict[k] = f"{val[:20]}... (length: {len(val)})"
    else:
        env_dict[k] = val if val else "❌ EMPTY"

st.json(env_dict)

# ══════════════════════════════════════════════════════════════
# 3. STREAMLIT SECRETS (st.secrets)
# ══════════════════════════════════════════════════════════════
st.header("3️⃣ Streamlit Secrets (st.secrets)")

try:
    all_secrets = dict(st.secrets)
    st.write(f"Total secrets found: {len(all_secrets)}")
    
    secrets_display = {}
    for k, v in all_secrets.items():
        if isinstance(v, str) and len(v) > 20:
            secrets_display[k] = f"{v[:20]}... (length: {len(v)})"
        else:
            secrets_display[k] = str(v)[:50]
    
    st.json(secrets_display)
except Exception as e:
    st.error(f"❌ Cannot read st.secrets: {e}")

# ══════════════════════════════════════════════════════════════
# 4. SUPABASE URL - ALL METHODS
# ══════════════════════════════════════════════════════════════
st.header("4️⃣ SUPABASE_URL - All Access Methods")

url_methods = {
    "st.secrets.get('SUPABASE_URL')": None,
    "st.secrets['SUPABASE_URL']": None,
    "os.getenv('SUPABASE_URL')": None,
    "os.environ.get('SUPABASE_URL')": None,
}

# Method 1
try:
    url_methods["st.secrets.get('SUPABASE_URL')"] = st.secrets.get("SUPABASE_URL", None)
except Exception as e:
    url_methods["st.secrets.get('SUPABASE_URL')"] = f"❌ Error: {e}"

# Method 2
try:
    url_methods["st.secrets['SUPABASE_URL']"] = st.secrets["SUPABASE_URL"]
except Exception as e:
    url_methods["st.secrets['SUPABASE_URL']"] = f"❌ Error: {e}"

# Method 3
url_methods["os.getenv('SUPABASE_URL')"] = os.getenv("SUPABASE_URL", None)

# Method 4
url_methods["os.environ.get('SUPABASE_URL')"] = os.environ.get("SUPABASE_URL", None)

for method, val in url_methods.items():
    if val and not str(val).startswith("❌"):
        st.success(f"✅ {method}: {val}")
    else:
        st.error(f"❌ {method}: {val}")

# ══════════════════════════════════════════════════════════════
# 5. SUPABASE KEY - ALL VARIANTS
# ══════════════════════════════════════════════════════════════
st.header("5️⃣ SUPABASE KEY - All Possible Variants")

key_variants = [
    "SUPABASE_KEY",
    "SUPABASE_ANON_KEY", 
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_PUBLISHABLE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "JWT_SECRET",
    "SUPABASE_JWT_SECRET",
]

found_keys = {}

for variant in key_variants:
    methods = {}
    
    # st.secrets.get
    try:
        val = st.secrets.get(variant, None)
        if val:
            methods["st.secrets.get"] = f"{str(val)[:20]}... (len: {len(val)})"
    except:
        pass
    
    # st.secrets direct
    try:
        val = st.secrets[variant]
        if val:
            methods["st.secrets[]"] = f"{str(val)[:20]}... (len: {len(val)})"
    except:
        pass
    
    # os.getenv
    val = os.getenv(variant, None)
    if val:
        methods["os.getenv"] = f"{str(val)[:20]}... (len: {len(val)})"
    
    # os.environ.get
    val = os.environ.get(variant, None)
    if val:
        methods["os.environ.get"] = f"{str(val)[:20]}... (len: {len(val)})"
    
    if methods:
        found_keys[variant] = methods

if found_keys:
    st.success(f"✅ Found {len(found_keys)} key variants:")
    st.json(found_keys)
else:
    st.error("❌ No Supabase keys found in ANY variant!")

# ══════════════════════════════════════════════════════════════
# 6. RECOMMENDED CONFIGURATION
# ══════════════════════════════════════════════════════════════
st.header("6️⃣ Recommended Configuration")

# Try the fallback pattern from working app
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))

st.code(f"""
# Current configuration result:
SUPABASE_URL = "{SUPABASE_URL}"
SUPABASE_KEY = "{SUPABASE_KEY[:20] if SUPABASE_KEY else 'EMPTY'}..." (length: {len(SUPABASE_KEY) if SUPABASE_KEY else 0})
""")

if SUPABASE_URL and SUPABASE_KEY:
    st.success("✅ Both URL and KEY are populated!")
    
    # Try to connect
    st.subheader("Testing Supabase Connection...")
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("✅ Supabase client created successfully!")
        
        # Try a simple query
        try:
            result = supabase.table("surveys").select("id").limit(1).execute()
            st.success(f"✅ Database connection works! Found {len(result.data)} surveys")
        except Exception as e:
            st.error(f"❌ Database query failed: {e}")
            
    except Exception as e:
        st.error(f"❌ Failed to create Supabase client: {e}")
else:
    st.error("❌ URL or KEY is missing!")
    
    if not SUPABASE_URL:
        st.warning("SUPABASE_URL is empty - check Vercel environment variables")
    if not SUPABASE_KEY:
        st.warning("SUPABASE_KEY is empty - check Vercel environment variables")

# ══════════════════════════════════════════════════════════════
# 7. VERCEL DEPLOYMENT INFO
# ══════════════════════════════════════════════════════════════
st.header("7️⃣ Vercel Deployment Information")

vercel_vars = {
    "VERCEL": os.getenv("VERCEL", "Not Vercel"),
    "VERCEL_ENV": os.getenv("VERCEL_ENV", "Unknown"),
    "VERCEL_URL": os.getenv("VERCEL_URL", "Unknown"),
    "VERCEL_GIT_COMMIT_SHA": os.getenv("VERCEL_GIT_COMMIT_SHA", "Unknown")[:8] + "...",
}

st.json(vercel_vars)

# ══════════════════════════════════════════════════════════════
# 8. ACTION ITEMS
# ══════════════════════════════════════════════════════════════
st.header("8️⃣ Next Steps")

st.markdown("""
### Based on the output above:

1. **Check Section 5** - Which key variant was found?
2. **Check Section 6** - Did connection test pass?
3. **If no keys found in Section 5:**
   - Go to Vercel → Environment Variables
   - Verify variables exist and are not empty
   - Check they're enabled for the correct environment (Production/Preview/Development)
   - Redeploy after any changes

4. **If keys found but connection fails:**
   - Verify the key is the correct type (anon key, not service_role)
   - Check Supabase project is not paused
   - Verify URL matches your Supabase project

5. **Copy ALL output above** and share for further diagnosis
""")

st.stop()
