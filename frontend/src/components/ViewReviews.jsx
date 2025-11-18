import React, { useState } from 'react';
import './ViewReviews.css';

const ViewReviews = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [courseId, setCourseId] = useState('');
  const [reviews, setReviews] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchReviews = async (searchCourseId) => {
    if (!searchCourseId.trim()) {
      setError('Please enter a course ID');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(
        `http://localhost:8000/api/reviews/${encodeURIComponent(searchCourseId.trim())}`
      );
      
      if (response.ok) {
        const data = await response.json();
        setReviews(data);
        if (data.total_reviews === 0) {
          setError('No reviews found for this course');
        }
      } else {
        setError('Failed to fetch reviews');
      }
    } catch (err) {
      console.error('Error fetching reviews:', err);
      setError('Error connecting to server');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    fetchReviews(courseId);
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <>
      {/* Floating Button */}
      <button
        className="view-reviews-floating-btn"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="View Course Reviews"
      >
        {isOpen ? '✕' : 'View'}
      </button>

      {/* Reviews Modal */}
      {isOpen && (
        <div className="reviews-modal-overlay" onClick={() => setIsOpen(false)}>
          <div className="reviews-modal" onClick={(e) => e.stopPropagation()}>
            <div className="reviews-header">
              <h3>View Course Reviews</h3>
              <button
                className="reviews-close-btn"
                onClick={() => setIsOpen(false)}
                aria-label="Close"
              >
                ✕
              </button>
            </div>

            {/* Search Section */}
            <div className="reviews-search-section">
              <form onSubmit={handleSearch}>
                <div className="search-input-group">
                  <input
                    type="text"
                    value={courseId}
                    onChange={(e) => setCourseId(e.target.value)}
                    placeholder="Enter Course ID (e.g., CS 7641)"
                    className="search-input"
                  />
                  <button type="submit" className="search-btn" disabled={loading}>
                    {loading ? '🔄' : '🔍'}
                  </button>
                </div>
              </form>

              {error && <div className="error-message">{error}</div>}
            </div>

            {/* Reviews Display */}
            {reviews && reviews.total_reviews > 0 && (
              <div className="reviews-content">
                {/* Aggregated Stats */}
                <div className="stats-card">
                  <h4>{reviews.course_id} - Review Summary</h4>
                  <div className="stats-grid">
                    <div className="stat-item">
                      <span className="stat-label">Reviews</span>
                      <span className="stat-value">{reviews.total_reviews}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Avg Difficulty</span>
                      <span className="stat-value">
                        {'★'.repeat(Math.round(reviews.avg_difficulty))}
                        {'☆'.repeat(5 - Math.round(reviews.avg_difficulty))}
                        <span className="stat-number">{reviews.avg_difficulty}/5</span>
                      </span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Avg Workload</span>
                      <span className="stat-value">{reviews.avg_workload} hrs/week</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Would Recommend</span>
                      <span className="stat-value recommend-stat">{reviews.recommend_percentage}%</span>
                    </div>
                  </div>
                </div>

                {/* Individual Reviews */}
                <div className="individual-reviews">
                  <h4>Student Reviews ({reviews.reviews.length})</h4>
                  {reviews.reviews.map((review, index) => (
                    <div key={index} className="review-card">
                      <div className="review-header">
                        <div className="review-metrics">
                          <span className="review-difficulty">
                            {'★'.repeat(review.difficulty)}
                            {'☆'.repeat(5 - review.difficulty)}
                          </span>
                          <span className="review-workload">{review.workload} hrs/week</span>
                          {review.would_recommend && (
                            <span className="recommend-badge">✓ Recommends</span>
                          )}
                        </div>
                        <span className="review-date">{formatDate(review.timestamp)}</span>
                      </div>
                      {review.review_text && (
                        <p className="review-text">{review.review_text}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default ViewReviews;
