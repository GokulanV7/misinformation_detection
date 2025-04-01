# rss_feed_ingestor.py
import feedparser
import re
import logging

logger = logging.getLogger(__name__)

def clean_html(raw_html):
    """Remove HTML tags from summary text."""
    clean_text = re.sub('<.*?>', '', raw_html)
    return clean_text.strip()

def get_rss_news(feed_url, max_articles=5):
    """Fetch and display news articles from an RSS feed with highlights."""
    logger.info(f"Attempting to fetch RSS feed from: {feed_url}") # Log URL being fetched
    articles = [] # Initialize empty list

    try:
        feed = feedparser.parse(feed_url)

        # --- Robustness Checks Added ---
        if feed.bozo: # Check if feedparser encountered potential errors
            bozo_msg = feed.bozo_exception.__class__.__name__ if hasattr(feed, 'bozo_exception') else 'Unknown error'
            logger.warning(f"Feedparser reported potential issues (bozo=1) for {feed_url}. Error type: {bozo_msg}")
            # Decide whether to proceed or return empty based on the error,
            # for now, we log and continue carefully.

        # Check if 'feed' attribute and 'title' exist before logging
        feed_title = "Unknown Title" # Default title
        if hasattr(feed, 'feed') and hasattr(feed.feed, 'title'):
             feed_title = feed.feed.title
        elif hasattr(feed, 'feed') and not hasattr(feed.feed, 'title'):
             logger.warning(f"Feed object exists for {feed_url}, but lacks a 'title'.")
        else:
            logger.warning(f"No 'feed' object found in parsed result for {feed_url}. Cannot determine title.")

        # Check if 'entries' exist and is a list
        if 'entries' in feed and isinstance(feed.entries, list) and len(feed.entries) > 0:
            logger.info(f"Fetched {len(feed.entries)} potential entries from {feed_title} ({feed_url})")
            count = 0
            for entry in feed.entries:
                if count >= max_articles:
                    break

                # Check if entry is valid and has essential components (title, link)
                if not isinstance(entry, dict):
                    logger.warning(f"Skipping invalid entry (not a dict) in feed: {feed_title}")
                    continue
                if not entry.get('title') or not entry.get('link'):
                    logger.warning(f"Skipping entry with missing title or link in feed: {feed_title}")
                    continue

                # Safely get summary
                summary = clean_html(entry.get('summary', 'No summary available'))

                # Construct the text
                full_text = f"Title: {entry.title}\nSummary: {summary}\nLink: {entry.link}"
                articles.append(full_text)
                count += 1
            logger.info(f"Successfully processed {len(articles)} valid articles from {feed_title}")

        else:
            logger.warning(f"No valid entries found in feed {feed_title} from {feed_url}. Check feed content and validity.")
            # feed.entries might exist but be empty []

    except Exception as e:
        logger.error(f"Unexpected error during RSS feed processing for {feed_url}: {e}", exc_info=True)
        # Return empty list on any unexpected error during processing

    return articles # Return potentially empty list


if __name__ == '__main__':
    # Example usage
    test_feeds = [
        "http://feeds.bbci.co.uk/news/rss.xml", # Usually works
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml", # Usually works
        "http://invalid.url.that.does.not.exist/rss.xml", # Should fail gracefully
        "https://www.google.com" # Not an RSS feed
    ]
    for url in test_feeds:
        print(f"\n--- Testing Feed: {url} ---")
        retrieved_articles = get_rss_news(url, max_articles=2)
        if retrieved_articles:
            for article in retrieved_articles:
                print(article + "\n---\n")
        else:
            print("No articles retrieved or processed.")
