import PropTypes from "prop-types";
import { Link } from "react-router-dom";
import { useState } from "react";

const BotMessage = ({ text, category, intent, ner, probabilities, id }) => {
  const [selectedResponse, setSelectedResponse] = useState(null);
  const [showOptions, setShowOptions] = useState(true);

  // Split the text into two responses (assuming they are separated by a delimiter like "||")
  const responses = text.split("||");

  const handleResponseSelection = (response) => {
    setSelectedResponse(response);
    setShowOptions(false);
  };

  return (
    <div className="text-left">
      <div className="bg-[#CAF0F8] p-4 rounded-lg inline-block space-y-4 text-black border-l-[8px] border-[#00137F]">
        {showOptions && responses.length > 1 ? (
          <div>
            <p>Category: {category}</p>
            <p>Intent: {intent}</p>
            <p>Named Entity Recognition: {ner}</p>
            <p>Please choose the best response:</p>
            {responses.map((response, index) => (
              <div key={index} className="mb-2">
                <button
                  onClick={() => handleResponseSelection(response)}
                  className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition duration-150"
                >
                  Response {index + 1}
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div>
            <p>Category: {category}</p>
            <p>Intent: {intent}</p>
            <p>Named Entity Recognition: {ner}</p>
            <p>{selectedResponse || text}</p>
          </div>
        )}
        <div className="pb-2">
          <Link
            to={{
              pathname: `/result/${id}`,
            }}
            className="text-gray-900 rounded-lg text-center hover:text-gray-600 transition duration-150 font-bold"
          >
            View More &gt;
          </Link>
        </div>
      </div>
    </div>
  );
};

BotMessage.propTypes = {
  text: PropTypes.string,
  category: PropTypes.string,
  intent: PropTypes.string,
  ner: PropTypes.string,
  id: PropTypes.number,
};

export default BotMessage;