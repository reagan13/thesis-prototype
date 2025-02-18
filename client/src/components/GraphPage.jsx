import { useData } from "../context/DataContext"; // Import the context

// Import images
import intentGraph from "../assets/intent-graph.png";
import categoryGraph from "../assets/category-graph.png";
import nerGraph from "../assets/ner-graph.png";
import overallGraph from "../assets/overall-graph.png";

const GraphPage = () => {
  const { graphType } = useData(); // Access graphType from the context
  const { isSidebarCollapsed } = useData(); // Access sidebar state from context

  // Determine the image path based on the graphType
  const getImagePath = (type) => {
    switch (type) {
      case "INTENT":
        return intentGraph;
      case "CATEGORY":
        return categoryGraph;
      case "NER":
        return nerGraph;
      case "OVERALL":
        return overallGraph;
      default:
        return ""; // Return empty if no matching type is found
    }
  };

  const imagePath = getImagePath(graphType);

  return (
    <div
      className=" ml-[300px] bg-black h-[670px] w-[1230px] overflow-y-auto p-4"
      style={{
        width: isSidebarCollapsed ? "1400px" : "1200px",
        marginLeft: isSidebarCollapsed ? "110px" : "300px",
        transition: "all 0.3s ease",
        scrollbarWidth: "none", // Firefox
        msOverflowStyle: "none", // Internet Explorer 10+
      }}
    >
      {/* Baseline Section */}
      <div
        className="mt-[10px] mb-10 ml-[40px] h-[610px] p-6 bg-[#133075d2] rounded-lg shadow-lg transition-transform transform hover:scale-105 hover:shadow-xl outline outline-2 outline-white outline-offset-2"
        style={{
          width: isSidebarCollapsed ? "1300px" : "1100px",
          transition: "all 0.3s ease",
        }}
      >
        <div className="text-2xl font-bold mb-2 text-white text-center pb-[20px]">
          {graphType}  Baseline 
        </div>
        <div className=" bg-[#CAF0F8] rounded-lg shadow-lg outline outline-2 outline-white outline-offset-2 p-4">
          <div>
            
            {imagePath ? (
              <img
                src={imagePath}
                alt={`${graphType} Baseline`}
                className="mt-2 h-[400px] w-full object-contain"
              />
            ) : (
              <p className="text-white">Image not found</p>
            )}
          </div>
        </div>
      </div>

      {/* Proposed Solution Section */}
      <div
        className=" ml-[40px] h-[610px] p-6 bg-[#133075d2] rounded-lg shadow-lg transition-transform transform hover:scale-105 hover:shadow-xl outline outline-2 outline-white outline-offset-2"
        style={{
          width: isSidebarCollapsed ? "1300px" : "1100px",
          transition: "all 0.3s ease",
        }}
      >
        <div className="text-2xl font-bold mb-2 text-white text-center pb-[20px]">
           {graphType} Proposed Solution 
        </div>
        <div className=" bg-[#CAF0F8] rounded-lg shadow-lg outline outline-2 outline-white outline-offset-2 p-4">
          <div>
            
            {imagePath ? (
              <img
                src={imagePath}
                alt={`${graphType} Proposed Solution`}
                className="mt-2 h-[400px] w-full object-contain"
              />
            ) : (
              <p className="text-white">Image not found</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GraphPage;
