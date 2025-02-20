import React, { useState } from "react";
import CustomTable from "../components/CustomTable";
import nerGraph from "../assets/ner-graph.png";

const Page = ({ name, categoryResponse, text, intentResponse, isSidebarCollapsed }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div
      className={
        "h-[610px] p-6 bg-white rounded-lg shadow-lg transition-transform transform hover:scale-105 hover:shadow-xl outline outline-2 outline-black outline-offset-2"
      }
      style={{
        width: isSidebarCollapsed ? "1250px" : "1120px",
        transition: "all 0.3s ease",
      }}
    >
      <div className="text-2xl font-bold mb-2 text-black text-center pb-[20px]">{name}</div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-white rounded-lg shadow-lg outline outline-2 outline-black outline-offset-2 p-4">
          <h2 className="text-lg font-semibold text-gray-800 border-b-2 border-blue-500 pb-1">
            Multi Task Classifications
          </h2>
          <div className="flex items-center justify-between mt-2">
            <span className="text-gray-700">NER</span>
            <span className="text-gray-700">Unknown</span>
          </div>
          <div className="flex items-center justify-between mt-1">
            <span className="text-gray-700">Category</span>
            <span className="text-gray-700">Unknown</span>
          </div>
          <div className="flex items-center justify-between mt-1">
            <span className="text-gray-700">Intent</span>
            <span className="text-gray-700">Unknown</span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg outline outline-2 outline-black outline-offset-2 p-4">
          <h2 className="text-lg font-semibold text-gray-800 pb-1">Analysis</h2>
          <CustomTable categoryResponse="unknown" intentResponse="unknown" />
        </div>
      </div>

      <div className="bg-white ml-[300px] rounded-lg shadow-lg outline outline-2 outline-black h-[260px] w-[500px] outline-offset-2 p-4 mb-8">
        <h2 className="pt-[4px] text-lg font-semibold text-black border-b-4 w-[466px] border-blue-500">Graph</h2>
        <div className="flex justify-center items-center gap-4 mt-4">
          <img
            src={nerGraph}
            alt="NER Graph"
            className="h-[180px] w-[500px] rounded-lg shadow-lg cursor-pointer"
            onClick={() => setIsModalOpen(true)}
          />
        </div>
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 flex justify-center items-center bg-white bg-opacity-50 z-50 p-4">
          <div className="bg-white p-4 rounded-lg shadow-lg relative">
            <button
              className="absolute top-2 right-2 bg-[#133075d2] text-white p-2 rounded-full"
              onClick={() => setIsModalOpen(false)}
            >
              &times;
            </button>
            <img src={nerGraph} alt="NER Graph" className="max-w-full max-h-screen rounded-lg" />
          </div>
        </div>
      )}
    </div>
  );
};

export default Page;
