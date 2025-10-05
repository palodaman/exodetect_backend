#!/usr/bin/env python3
"""
Export logged queries as training data for model retraining
Can be run as standalone script or imported as module
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from database import init_db, close_db
from database.schema import Query, QueryType
from dotenv import load_dotenv


class TrainingDataExporter:
    """Export logged queries as training data"""

    def __init__(self):
        self.queries = []

    async def fetch_queries(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        query_types: Optional[List[QueryType]] = None,
        organization_id: Optional[str] = None,
        min_confidence: Optional[str] = None
    ) -> List[Query]:
        """
        Fetch queries from MongoDB

        Args:
            start_date: Filter from this date
            end_date: Filter to this date
            query_types: Filter by query types
            organization_id: Filter by organization
            min_confidence: Minimum confidence level (High, Medium, Low)

        Returns:
            List of Query documents
        """
        # Build filter
        filter_conditions = []

        if start_date:
            filter_conditions.append(Query.created_at >= start_date)

        if end_date:
            filter_conditions.append(Query.created_at <= end_date)

        if query_types:
            filter_conditions.append(Query.query_type.in_(query_types))

        if organization_id:
            filter_conditions.append(Query.organization_id == organization_id)

        # Execute query
        query_builder = Query.find()

        if filter_conditions:
            query_builder = query_builder.find(*filter_conditions)

        queries = await query_builder.sort("-created_at").to_list()

        # Filter by confidence if specified
        if min_confidence:
            confidence_order = {"High": 3, "Medium": 2, "Low": 1}
            min_level = confidence_order.get(min_confidence, 0)

            queries = [
                q for q in queries
                if confidence_order.get(
                    q.prediction_result.get("confidence", "Low"),
                    0
                ) >= min_level
            ]

        self.queries = queries
        return queries

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export queries to pandas DataFrame suitable for training

        Returns:
            DataFrame with features and labels
        """
        if not self.queries:
            return pd.DataFrame()

        rows = []

        for query in self.queries:
            # Extract features from input_data and prediction_result
            features = query.input_data.copy()

            # Add prediction result features if available
            pred_result = query.prediction_result

            if "features" in pred_result:
                features.update(pred_result["features"])

            # Add metadata
            features["query_type"] = query.query_type
            features["model_version"] = query.model_version
            features["timestamp"] = query.created_at

            # Add label (ground truth)
            # For now, we use the model's prediction as the label
            # In production, you'd want human verification
            features["label"] = 1 if pred_result.get("model_label") == "CANDIDATE" else 0
            features["probability"] = pred_result.get("model_probability_candidate", 0.5)

            rows.append(features)

        return pd.DataFrame(rows)

    def export_to_csv(self, output_path: str) -> str:
        """
        Export to CSV file

        Args:
            output_path: Path to output CSV file

        Returns:
            Path to created file
        """
        df = self.export_to_dataframe()

        if df.empty:
            print("⚠️  No data to export")
            return None

        df.to_csv(output_path, index=False)
        print(f"✅ Exported {len(df)} records to {output_path}")

        return output_path

    def export_light_curves(self, output_dir: str) -> List[str]:
        """
        Export light curve data as separate numpy files

        Args:
            output_dir: Directory to save light curve files

        Returns:
            List of created file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        created_files = []

        for i, query in enumerate(self.queries):
            # Check if query has light curve data
            input_data = query.input_data

            if "time" in input_data and "flux" in input_data:
                # Extract light curve
                time = np.array(input_data["time"])
                flux = np.array(input_data["flux"])
                flux_err = np.array(input_data.get("flux_err", []))

                # Get label
                label = 1 if query.prediction_result.get("model_label") == "CANDIDATE" else 0

                # Save as npz file
                filename = f"lc_{i:06d}_label{label}_{query.query_type}.npz"
                filepath = os.path.join(output_dir, filename)

                np.savez(
                    filepath,
                    time=time,
                    flux=flux,
                    flux_err=flux_err,
                    label=label,
                    probability=query.prediction_result.get("model_probability_candidate", 0.5),
                    query_id=str(query.id),
                    timestamp=query.created_at.isoformat()
                )

                created_files.append(filepath)

        print(f"✅ Exported {len(created_files)} light curves to {output_dir}")
        return created_files

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about exported data"""
        if not self.queries:
            return {}

        total = len(self.queries)

        # Count by query type
        by_type = {}
        for q in self.queries:
            by_type[q.query_type] = by_type.get(q.query_type, 0) + 1

        # Count by label
        candidates = sum(
            1 for q in self.queries
            if q.prediction_result.get("model_label") == "CANDIDATE"
        )

        # Count by confidence
        by_confidence = {}
        for q in self.queries:
            conf = q.prediction_result.get("confidence", "Unknown")
            by_confidence[conf] = by_confidence.get(conf, 0) + 1

        # Date range
        dates = [q.created_at for q in self.queries]
        date_range = {
            "start": min(dates).isoformat() if dates else None,
            "end": max(dates).isoformat() if dates else None
        }

        return {
            "total_queries": total,
            "candidates": candidates,
            "non_candidates": total - candidates,
            "by_query_type": by_type,
            "by_confidence": by_confidence,
            "date_range": date_range
        }


async def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export logged queries as training data"
    )
    parser.add_argument(
        "--output",
        default=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--light-curves",
        help="Export light curves to this directory"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Export queries from last N days (default: 7)"
    )
    parser.add_argument(
        "--min-confidence",
        choices=["High", "Medium", "Low"],
        help="Minimum confidence level"
    )
    parser.add_argument(
        "--organization-id",
        help="Filter by organization ID"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    print("ExoDetect Training Data Exporter")
    print("=" * 50)

    # Initialize database
    print("\n📊 Connecting to database...")
    await init_db()

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")

    # Create exporter
    exporter = TrainingDataExporter()

    # Fetch queries
    print(f"🔍 Fetching queries...")
    queries = await exporter.fetch_queries(
        start_date=start_date,
        end_date=end_date,
        organization_id=args.organization_id,
        min_confidence=args.min_confidence
    )

    print(f"✅ Found {len(queries)} queries")

    if len(queries) == 0:
        print("⚠️  No queries found. Nothing to export.")
        await close_db()
        return

    # Show statistics
    stats = exporter.get_statistics()
    print("\n📈 Statistics:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Candidates: {stats['candidates']}")
    print(f"   Non-candidates: {stats['non_candidates']}")
    print(f"   By type: {stats['by_query_type']}")
    print(f"   By confidence: {stats['by_confidence']}")

    # Export to CSV
    print(f"\n💾 Exporting to {args.output}...")
    exporter.export_to_csv(args.output)

    # Export light curves if requested
    if args.light_curves:
        print(f"\n💾 Exporting light curves to {args.light_curves}...")
        exporter.export_light_curves(args.light_curves)

    # Close database
    await close_db()

    print("\n✅ Export complete!")


if __name__ == "__main__":
    asyncio.run(main())
