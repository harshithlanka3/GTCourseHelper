import React, { useState } from 'react';
import './FeedbackButton.css';

const FeedbackButton = () => {
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [courseId, setCourseId] = useState('');
  const [difficulty, setDifficulty] = useState(3);
  const [workload, setWorkload] = useState(10);
  const [wouldRecommend, setWouldRecommend] = useState(true);
  const [reviewText, setReviewText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!courseId.trim()) {
      setSubmitMessage('Please enter a course ID');
      return;
    }

    setIsSubmitting(true);
    setSubmitMessage('');

    try {
      const response = await fetch('http://localhost:8000/api/reviews', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          course_id: courseId.trim(),
          difficulty,
          workload,
          would_recommend: wouldRecommend,
          review_text: reviewText.trim() || null,
        }),
      });

      if (response.ok) {
        setSubmitMessage('✓ Review submitted successfully!');
        // Reset form
        setCourseId('');
        setDifficulty(3);
        setWorkload(10);
        setWouldRecommend(true);
        setReviewText('');
        // Close form after 2 seconds
        setTimeout(() => {
          setIsFormOpen(false);
          setSubmitMessage('');
        }, 2000);
      } else {
        setSubmitMessage('Failed to submit review. Please try again.');
      }
    } catch (error) {
      console.error('Error submitting review:', error);
      setSubmitMessage('Error submitting review. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      {/* Floating Button */}
      <button
        className="feedback-floating-btn"
        onClick={() => setIsFormOpen(!isFormOpen)}
        aria-label="Submit Course Feedback"
      >
        {isFormOpen ? '✕' : 'Review'}
      </button>

      {/* Feedback Form Modal */}
      {isFormOpen && (
        <div className="feedback-modal-overlay" onClick={() => setIsFormOpen(false)}>
          <div className="feedback-modal" onClick={(e) => e.stopPropagation()}>
            <div className="feedback-header">
              <h3>Submit Course Feedback</h3>
              <button
                className="feedback-close-btn"
                onClick={() => setIsFormOpen(false)}
                aria-label="Close"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleSubmit} className="feedback-form">
              <div className="form-group">
                <label htmlFor="courseId">
                  Course ID <span className="required">*</span>
                </label>
                <input
                  id="courseId"
                  type="text"
                  value={courseId}
                  onChange={(e) => setCourseId(e.target.value)}
                  placeholder="e.g., CS 7641"
                  required
                />
              </div>

              <div className="form-group">
                <label>
                  Difficulty: {difficulty}/5
                </label>
                <div className="star-rating">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <span
                      key={star}
                      className={`star ${star <= difficulty ? 'filled' : ''}`}
                      onClick={() => setDifficulty(star)}
                    >
                      ★
                    </span>
                  ))}
                </div>
              </div>

              <div className="form-group">
                <label>
                  Workload: {workload} hrs/week
                </label>
                <input
                  type="range"
                  min="1"
                  max="40"
                  value={workload}
                  onChange={(e) => setWorkload(parseInt(e.target.value))}
                  className="workload-slider"
                />
              </div>

              <div className="form-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={wouldRecommend}
                    onChange={(e) => setWouldRecommend(e.target.checked)}
                  />
                  I would recommend this course
                </label>
              </div>

              <div className="form-group">
                <label htmlFor="reviewText">
                  Your Review (optional)
                </label>
                <textarea
                  id="reviewText"
                  value={reviewText}
                  onChange={(e) => setReviewText(e.target.value)}
                  placeholder="Share your experience with this course..."
                  rows="4"
                  maxLength="500"
                />
                <span className="char-count">{reviewText.length}/500</span>
              </div>

              {submitMessage && (
                <div className={`submit-message ${submitMessage.includes('✓') ? 'success' : 'error'}`}>
                  {submitMessage}
                </div>
              )}

              <div className="form-actions">
                <button
                  type="button"
                  onClick={() => setIsFormOpen(false)}
                  className="btn-cancel"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="btn-submit"
                >
                  {isSubmitting ? 'Submitting...' : 'Submit Review'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default FeedbackButton;
