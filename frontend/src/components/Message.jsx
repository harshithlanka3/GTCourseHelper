import React from 'react'
import ReactMarkdown from 'react-markdown'
import './Message.css'

const parseCourseRecommendations = (text) => {
  if (!text) {
    return []
  }

  const normalized = text.trim()
  if (!/^\d+\.\s+\*\*/m.test(normalized)) {
    return []
  }

  const blocks = normalized
    .split(/\n(?=\d+\.\s+\*\*)/g)
    .map((block) => block.trim())
    .filter(Boolean)

  const courses = []

  blocks.forEach((block) => {
    const headerMatch = block.match(/^\d+\.\s+\*\*(.+?)\*\*/)
    if (!headerMatch) {
      return
    }

    const header = headerMatch[1].trim()
    const [rawCourseId, ...titleParts] = header.split(':')
    const courseId = (rawCourseId || '').trim()
    const title = titleParts.join(':').trim()

    const rationaleMatch = block.match(/Rationale:\s*([\s\S]*?)\nSections:/i)
    const sectionsMatch = block.match(/Sections:\s*([\s\S]*?)\nPrerequisites:/i)
    const prerequisitesMatch = block.match(/Prerequisites:\s*([\s\S]*?)\nConfidence:/i)
    const confidenceMatch = block.match(/Confidence:\s*([^\n]+)/i)

    if (!rationaleMatch || !sectionsMatch || !prerequisitesMatch) {
      return
    }

    courses.push({
      courseId,
      title: title || courseId,
      rationale: rationaleMatch[1].trim(),
      sections: sectionsMatch[1].trim(),
      prerequisites: prerequisitesMatch[1].trim(),
      confidence: confidenceMatch ? confidenceMatch[1].trim() : '',
    })
  })

  return courses
}

const toList = (value) => {
  if (!value) {
    return []
  }

  if (value.includes(';')) {
    return value.split(';').map((item) => item.trim()).filter(Boolean)
  }

  return [value.trim()].filter(Boolean)
}

function Message({ message }) {
  const isUser = message.role === 'user'
  const recommendations = !isUser ? parseCourseRecommendations(message.content) : []

  const sortedRecommendations = React.useMemo(() => {
    if (!recommendations || recommendations.length === 0) {
      return []
    }

    const confidenceOrder = {
      high: 0,
      medium: 1,
      low: 2,
    }

    return recommendations
      .map((course, index) => ({ course, index }))
      .sort((aEntry, bEntry) => {
        const aConf = (aEntry.course.confidence || '').toLowerCase()
        const bConf = (bEntry.course.confidence || '').toLowerCase()
        const aRank = confidenceOrder[aConf] ?? 3
        const bRank = confidenceOrder[bConf] ?? 3

        if (aRank !== bRank) {
          return aRank - bRank
        }

        return aEntry.index - bEntry.index
      })
      .map((entry) => entry.course)
  }, [recommendations])

  return (
    <div className={`Message Message-${message.role}`}>
      <div className="Message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="Message-content">
        {isUser ? (
          <p>{message.content}</p>
        ) : sortedRecommendations.length > 0 ? (
          <div className="CourseRecommendations">
            {sortedRecommendations.map((course, index) => {
              const sectionsList = toList(course.sections)
              const confidenceClass = course.confidence
                ? `CourseCard-confidence-${course.confidence.toLowerCase().replace(/[^a-z]/g, '')}`
                : ''

              return (
                <div key={`${course.courseId}-${index}`} className="CourseCard">
                  <div className="CourseCard-header">
                    <span className="CourseCard-code">{course.courseId}</span>
                    {course.confidence && (
                      <span className={`CourseCard-confidence ${confidenceClass}`}>
                        {course.confidence}
                      </span>
                    )}
                  </div>
                  <h3 className="CourseCard-title">{course.title}</h3>
                  <p className="CourseCard-rationale">{course.rationale}</p>
                  <div className="CourseCard-detail">
                    <span className="CourseCard-label">Sections</span>
                    {sectionsList.length > 1 ? (
                      <ul className="CourseCard-list">
                        {sectionsList.map((item, itemIndex) => (
                          <li key={itemIndex}>{item}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="CourseCard-text">{course.sections}</p>
                    )}
                  </div>
                  <div className="CourseCard-detail">
                    <span className="CourseCard-label">Prerequisites</span>
                    <p className="CourseCard-text">{course.prerequisites}</p>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="Message-markdown">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}

export default Message

