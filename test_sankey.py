"""
Quick test script to verify the Sankey chart generation works correctly.
Run with: python manage.py shell < test_sankey.py
"""

from apps.charts.patients.production_age_charts import (
    build_enhanced_age_referral_sankey,
    build_simplified_age_bar_chart,
)
from apps.core.models import Patients, Referrals

print("Testing simplified age bar chart...")
try:
    chart_html = build_simplified_age_bar_chart("dark")
    if chart_html and len(chart_html) > 100:
        print("✅ Simplified age bar chart generated successfully")
        print(f"   Length: {len(chart_html)} characters")
    else:
        print("⚠️  Chart HTML seems too short or empty")
except Exception as e:
    print(f"❌ Error generating age bar chart: {e}")

print("\nTesting enhanced age → referral Sankey...")
try:
    sankey_html = build_enhanced_age_referral_sankey("dark")
    if sankey_html and len(sankey_html) > 100:
        print("✅ Enhanced Sankey diagram generated successfully")
        print(f"   Length: {len(sankey_html)} characters")
    else:
        print("⚠️  Sankey HTML seems too short or empty")
except Exception as e:
    print(f"❌ Error generating Sankey: {e}")

print("\nChecking data availability...")

patient_count = Patients.objects.count()
referral_count = Referrals.objects.count()

print(f"   Patients in database: {patient_count}")
print(f"   Referrals in database: {referral_count}")

if patient_count == 0:
    print("⚠️  No patients found - charts will show 'No data available'")
if referral_count == 0:
    print("⚠️  No referrals found - Sankey will show 'No referral data available'")

print("\n✅ All tests completed!")
