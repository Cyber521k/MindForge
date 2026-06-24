"""In-CLI review prompt for training entries."""

import sys
import os
import logging

logger = logging.getLogger(__name__)


def review_session(db, limit=100):
    """Run an interactive review session in the CLI.

    Args:
        db: Database instance
        limit: Maximum number of entries to review

    Returns:
        dict with review stats
    """
    entries = db.get_pending_entries(limit=limit)

    if not entries:
        print("\nNo pending training entries to review.")
        return {"reviewed": 0, "accepted": 0, "rejected": 0, "edited": 0, "skipped": 0}

    stats = {"reviewed": 0, "accepted": 0, "rejected": 0, "edited": 0, "skipped": 0}

    print(f"\n{'='*60}")
    print(f"  Review Session - {len(entries)} pending entries")
    print(f"{'='*60}\n")

    for entry in entries:
        stats["reviewed"] += 1

        print(f"--- Entry {entry['id']} ---")
        print(f"Subject: {entry.get('subject', 'N/A')}")
        print(f"\nPrompt:")
        print(f"  {entry['prompt'][:500]}{'...' if len(entry['prompt']) > 500 else ''}")
        print(f"\nChosen (correct):")
        print(f"  {entry['chosen']}")
        print(f"\nRejected (model's wrong answer):")
        print(f"  {entry['rejected']}")
        print()

        while True:
            print("[A] Accept  [R] Reject  [E] Edit  [S] Skip")
            try:
                choice = input("> ").strip().lower()
            except EOFError:
                choice = "s"

            if choice == "a":
                db.update_entry_status(entry["id"], "accepted")
                db.store_review_session(entry["id"], "accept")
                stats["accepted"] += 1
                print("  -> Accepted\n")
                break
            elif choice == "r":
                db.update_entry_status(entry["id"], "rejected")
                db.store_review_session(entry["id"], "reject")
                stats["rejected"] += 1
                print("  -> Rejected\n")
                break
            elif choice == "e":
                print("  Enter new chosen text (or empty to keep current):")
                try:
                    new_chosen = input("  chosen> ").strip()
                except EOFError:
                    new_chosen = ""
                if new_chosen:
                    db.update_training_entry(entry["id"], chosen=new_chosen)

                print("  Enter new rejected text (or empty to keep current):")
                try:
                    new_rejected = input("  rejected> ").strip()
                except EOFError:
                    new_rejected = ""
                if new_rejected:
                    db.update_training_entry(entry["id"], rejected=new_rejected)

                db.update_entry_status(entry["id"], "edited")
                db.store_review_session(
                    entry["id"], "edit",
                    edited_chosen=new_chosen or None,
                    edited_rejected=new_rejected or None,
                )
                stats["edited"] += 1
                print("  -> Edited\n")
                break
            elif choice == "s":
                db.update_entry_status(entry["id"], "skipped")
                db.store_review_session(entry["id"], "skip")
                stats["skipped"] += 1
                print("  -> Skipped\n")
                break
            else:
                print("  Invalid choice. Please enter A, R, E, or S.")

    print(f"\n{'='*60}")
    print(f"  Review Complete")
    print(f"{'='*60}")
    print(f"  Reviewed: {stats['reviewed']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Edited:   {stats['edited']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"{'='*60}\n")

    return stats
